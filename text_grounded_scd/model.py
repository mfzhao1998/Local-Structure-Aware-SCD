from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from sam3.model.vitdet import ViT
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer


DEFAULT_CLASS_NAMES = [
    "no-change",
    "water",
    "ground",
    "low vegetation",
    "tree",
    "building",
    "playground",
]
DEFAULT_BPE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sam3"
    / "assets"
    / "bpe_simple_vocab_16e6.txt.gz"
)

class CNNBasedSpatialBranch(nn.Module):
    """CNN-based spatial branch of the S2CE encoder."""

    def __init__(self):
        super().__init__()
        self.stage_h4 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, bias=False))
        self.stage_h8 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x_h4 = self.stage_h4(x)
        x_h8 = self.stage_h8(x_h4)
        return x_h4, x_h8
class GlobalSemanticInjection(nn.Module):
    """Global semantic injection stage of DTSA."""

    def __init__(self, vis_dim, text_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.vis_proj = nn.Conv2d(vis_dim, hidden_dim, 1)
        self.text_proj = nn.Linear(text_dim, hidden_dim) 
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, vision, text):
        B, C, H, W = vision.shape
        vis_feat = self.vis_proj(vision)
        q = vis_feat.flatten(2).transpose(1, 2)
        text_feat = self.text_proj(text)
        k = v = text_feat.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=q, key=k, value=v, need_weights=True)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        if not self.training:
            self.last_cross_feat = x.detach().cpu()
        x = self.final_proj(x)
        return x

class ChannelModulation(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelModulation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialModulation(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialModulation, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ChannelReweighting(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class ScaleWiseCueAggregation(nn.Module):
    """Scale-wise cue aggregation used by the CSGD decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.res_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.attn = ChannelReweighting(out_channels)

    def forward(self, x):
        x = self.bottleneck(x)
        identity = x
        out = self.res_conv(x)
        out = self.attn(out) 
        return out + identity 

class DualAxisFeatureRefinement(nn.Module):
    """Class-score-conditioned channel and spatial refinement in CSGD."""

    def __init__(self, in_channels, out_channels):
        super(DualAxisFeatureRefinement, self).__init__()
        self.conv_mix = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelModulation(out_channels)
        self.sa = SpatialModulation(kernel_size=7)
        self.proj = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x_feat, s1, s2, s_diff):
        combined = torch.cat([x_feat, s1, s2, s_diff], dim=1)
        feat = self.conv_mix(combined)
        feat = feat * self.ca(feat)
        spat_map = self.sa(feat) 
        if s_diff.shape[1] > 1:
             s_diff_attn = torch.mean(s_diff, dim=1, keepdim=True)
        else:
             s_diff_attn = s_diff
        final_spatial_weight = torch.sigmoid(spat_map + s_diff_attn) 
        if not self.training:
            self.last_spatial_attention = final_spatial_weight[0, 0].detach().cpu()
        feat = feat * final_spatial_weight
        return self.bn(self.proj(feat))

class DecoderBlock(nn.Module):
    """Top-down decoder block in the multi-cue multiscale decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x_deep, x_skip=None):
        x_deep = self.up(x_deep)
        if x_skip is not None:
            if x_deep.shape[-2:] != x_skip.shape[-2:]:
                x_deep = F.interpolate(x_deep, size=x_skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x_skip, x_deep], dim=1)
        else:
            x = x_deep
        return self.conv_block(x)

class LightweightAdapter(nn.Module):
    """Bottleneck adapter inserted before a frozen Transformer block."""

    def __init__(self, blk) -> None:
        super(LightweightAdapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        prompted = x + prompt
        net = self.block(prompted)
        return net
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)

class DynamicLocalAlignment(nn.Module):
    """Dynamic local alignment stage of DTSA."""

    def __init__(self, word_dim=256, vision_dim=256, kernel_size=3):
        super().__init__()
        self.vision_dim = vision_dim
        self.kernel_size = kernel_size
        self.out_dim = self.vision_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, self.out_dim)
        self.logit_scale = nn.Parameter(torch.ones(1)*2.0)
        nn.init.normal_(self.txt.weight, mean=0, std=0.001)
        nn.init.constant_(self.txt.bias, 0)
    
    def kernel_normalizer(self, mask0, kernel):
        n, mask_c = mask0.size()
        mask_channel = int(mask_c / float(kernel**2))
        mask = mask0.view(n, mask_channel, -1)
        mask = F.softmax(mask, dim=-1, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel)
        mask_sum = mask.sum(dim=(-1, -2), keepdims=True) + 1e-6
        mask = mask / mask_sum 
        return mask

    def forward(self, vision, text):
        batch_size, _, _, _ = vision.size()
        text_feats = self.txt(text)
        weight_raw, bias = text_feats[:, :-1], text_feats[:, -1]
        weight = self.kernel_normalizer(weight_raw, self.kernel_size)
        vision_norm = F.normalize(vision, dim=1)
        alignment_outputs = []
        for batch_index in range(batch_size):
            output = F.conv2d(
                vision_norm[batch_index].unsqueeze(0),
                weight,
                bias=bias,
                padding=self.kernel_size // 2,
            )
            alignment_outputs.append(output * self.logit_scale.exp())
        return torch.cat(alignment_outputs, dim=0)
    
def _create_sam3_semantic_branch(img_size):
    """Create the SAM 3-based semantic branch of S2CE."""
    return ViT(
        img_size=img_size,
        pretrain_img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=16,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
    )

class TextGroundedSCD(nn.Module):
    """Local-structure-aware text-grounded SCD model from the paper."""

    LEGACY_MODULE_PREFIXES = {
        "sam3_vit.": "sam3_semantic_branch.",
        "sem_fusion.": "dual_axis_refinement.",
        "local_align.": "dynamic_local_alignment.",
        "reduce1.": "scale_aggregation1.",
        "reduce2.": "scale_aggregation2.",
        "reduce3.": "scale_aggregation3.",
        "reduce4.": "scale_aggregation4.",
        "up1.": "decoder_block1.",
        "up2.": "decoder_block2.",
        "up3.": "decoder_block3.",
        "head.": "binary_change_head.",
        "sem.": "global_semantic_injection.",
        "detail_exporter.": "cnn_spatial_branch.",
    }

    def __init__(self, 
                 checkpoint_path=None, 
                 bpe_path=None,
                 img_size=512,
                 num_classes=7,
                 class_names=None) -> None: 
        super().__init__()
        bpe_path = Path(bpe_path) if bpe_path is not None else DEFAULT_BPE_PATH
        self.sam3_semantic_branch = _create_sam3_semantic_branch(img_size)
        self.class_names = class_names if class_names is not None else DEFAULT_CLASS_NAMES
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path)
            if 'model' in ckpt:
                ckpt = ckpt['model']
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            new_ckpt = dict()
            for k, v in ckpt.items():
                prefix = "detector.backbone.vision_backbone.trunk."
                if prefix.rstrip(".") in k and 'freqs_cis' not in k:
                    new_ckpt[k[len(prefix):]] = v
            if 'patch_embed.proj.weight' in new_ckpt:
                patch_ckpt = new_ckpt['patch_embed.proj.weight'] 
                patch_model = self.sam3_semantic_branch.patch_embed.proj.weight 
                if patch_ckpt.shape != patch_model.shape:
                    new_ckpt['patch_embed.proj.weight'] = F.interpolate(
                        patch_ckpt, 
                        size=patch_model.shape[-2:], 
                        mode='bicubic', 
                        align_corners=False
                    )
            if 'pos_embed' in new_ckpt:
                pos_embed_checkpoint = new_ckpt['pos_embed'] 
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches_checkpoint = pos_embed_checkpoint.shape[1] - 1
                orig_size = int(num_patches_checkpoint ** 0.5)
                num_patches_current = self.sam3_semantic_branch.pos_embed.shape[1] - 1
                new_size = int(num_patches_current ** 0.5) 
                if orig_size != new_size:
                    extra_tokens = pos_embed_checkpoint[:, :1] 
                    pos_tokens = pos_embed_checkpoint[:, 1:]
                    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    new_ckpt['pos_embed'] = new_pos_embed
            self.sam3_semantic_branch.load_state_dict(new_ckpt, strict=False)


        for param in self.sam3_semantic_branch.parameters():
            param.requires_grad = False
        
        blocks = []
        for block in self.sam3_semantic_branch.blocks:
            blocks.append(LightweightAdapter(block))  

        if not bpe_path.exists():
            print(f"BPE file not found at {bpe_path}")
            self.tokenizer = None
            self.text_encoder = None
        else:
            self.tokenizer = SimpleTokenizer(bpe_path)
            self.text_encoder = VETextEncoder(
                d_model=256,
                tokenizer=self.tokenizer,
                context_length=32,
                vocab_size=self.tokenizer.vocab_size,
                use_ln_post=True,
                width=1024,
                layers=24,
            )
            if checkpoint_path:
                text_prefix = "detector.backbone.language_backbone."
                text_encoder_dict = {}
                model_state_dict = self.text_encoder.state_dict()

                for k, v in ckpt.items():
                    if k.startswith(text_prefix):
                        new_key = k[len(text_prefix):]
                        if (
                            new_key in model_state_dict
                            and v.shape == model_state_dict[new_key].shape
                        ):
                            text_encoder_dict[new_key] = v

                if text_encoder_dict:
                    self.text_encoder.load_state_dict(
                        text_encoder_dict, strict=False
                    )
                    print(
                        "Text Encoder loaded: "
                        f"{len(text_encoder_dict)} keys matched."
                    )
                else:
                    print("Warning: No Text Encoder weights loaded from checkpoint.")

            self.text_encoder.float()
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        self.multiscale_semantic_features=[]
        def hook_fn(module, input, output):
            self.multiscale_semantic_features.append(output)
        semantic_stage_indices = [7, 15, 23, 31]
        for idx in semantic_stage_indices:
            self.sam3_semantic_branch.blocks[idx].register_forward_hook(hook_fn)

        self.sam3_semantic_branch.blocks = nn.Sequential(*blocks)
        fusion_in_channels = 512 + num_classes * 3
        self.dual_axis_refinement = DualAxisFeatureRefinement(in_channels=fusion_in_channels, out_channels=512)
        self.dynamic_local_alignment = DynamicLocalAlignment(word_dim=256, vision_dim=512, kernel_size=7)
        self.scale_aggregation1 = ScaleWiseCueAggregation(1024*3+num_classes+256, 512) 
        self.scale_aggregation2 = ScaleWiseCueAggregation(1024*3+num_classes+256, 512)
        self.scale_aggregation3 = ScaleWiseCueAggregation(1024*3+num_classes, 512)
        self.scale_aggregation4 = ScaleWiseCueAggregation(1024*3+num_classes, 512)
        self.decoder_block1 = DecoderBlock(1024, 512)
        self.decoder_block2 = DecoderBlock(1024, 512)
        self.decoder_block3 = DecoderBlock(1024, 512)
        self.binary_change_head = nn.Conv2d(512, 1, 1)
        self.global_semantic_injection = GlobalSemanticInjection(vis_dim=2048, text_dim=256, hidden_dim=512)
        self._cached_text_embedding = None
        self.cnn_spatial_branch = CNNBasedSpatialBranch()
        

    def load_state_dict(self, state_dict, strict=True, assign=False):
        renamed_state_dict = state_dict.__class__()
        for key, value in state_dict.items():
            renamed_key = key
            for legacy_prefix, paper_prefix in self.LEGACY_MODULE_PREFIXES.items():
                if key.startswith(legacy_prefix):
                    renamed_key = paper_prefix + key[len(legacy_prefix):]
                    break
            renamed_state_dict[renamed_key] = value
        return super().load_state_dict(
            renamed_state_dict, strict=strict, assign=assign
        )

    def extract_multiscale_semantic_features(self, x):
        self.multiscale_semantic_features = []
        _ = self.sam3_semantic_branch(x)
        processed_features = []
        for feature in self.multiscale_semantic_features:
            if feature.dim() == 4 and feature.shape[-1] == 1024:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            processed_features.append(feature)
        return processed_features
        
    def get_text_embeddings(self, device):
        if (
            self._cached_text_embedding is not None
            and self._cached_text_embedding.device == device
        ):
            return self._cached_text_embedding
        prompts = self.class_names
        tokenized = self.tokenizer(prompts, context_length=32).to(device) 
        with torch.no_grad():
            outputs = self.text_encoder(prompts, device=device)
            text_memory = outputs[1].transpose(0, 1)
            eot_indices = tokenized.argmax(dim=-1)
            text_emb = text_memory[torch.arange(text_memory.shape[0]), eot_indices]
            text_emb = F.normalize(text_emb, p=2, dim=-1, eps=1e-6)
        self._cached_text_embedding = text_emb
        return text_emb
    
    def forward(self, t1, t2=None):
        B, C, H, W = t1.shape
        if t2 is None:
            raise ValueError("error: t2 input is required")

        # Semantic-Spatial Complementary Encoder (S2CE)
        semantic_features_t1 = self.extract_multiscale_semantic_features(t1)
        semantic_features_t2 = self.extract_multiscale_semantic_features(t2)
        spatial_t1_h4, spatial_t1_h8 = self.cnn_spatial_branch(t1)
        spatial_t2_h4, spatial_t2_h8 = self.cnn_spatial_branch(t2)

        # Dynamic Text-Spatial Aligner (DTSA)
        text_emb=self.get_text_embeddings(t1.device)
        dynamic_local_alignment = self.dynamic_local_alignment
        text_emb = text_emb.to(dtype=torch.float32)

        category_aware_t1 = torch.cat([semantic_features_t1[2], semantic_features_t1[3]], dim=1)
        category_aware_t1 = self.global_semantic_injection(category_aware_t1, text_emb)
        class_scores_t1 = dynamic_local_alignment(category_aware_t1, text_emb)
        class_scores_t1_full = F.interpolate(class_scores_t1, size=(H, W), mode='bilinear', align_corners=False)
        category_aware_t2 = torch.cat([semantic_features_t2[2], semantic_features_t2[3]], dim=1)
        category_aware_t2 = self.global_semantic_injection(category_aware_t2, text_emb)
        class_scores_t2 = dynamic_local_alignment(category_aware_t2, text_emb)
        class_scores_t2_full = F.interpolate(class_scores_t2, size=(H, W), mode='bilinear', align_corners=False)
        class_score_difference = torch.abs(F.softmax(class_scores_t1, dim=1) - F.softmax(class_scores_t2, dim=1))

        # Class-Score-Guided Decoder (CSGD): scale-wise cue aggregation
        difference_stage4 = torch.abs(semantic_features_t1[3] - semantic_features_t2[3])
        cues_stage4=torch.cat([semantic_features_t1[3], semantic_features_t2[3], difference_stage4, class_score_difference], dim=1)
        aggregated_stage4=self.scale_aggregation4(cues_stage4)
        aggregated_stage4=F.interpolate(aggregated_stage4, size=(H//16, W//16), mode='bilinear')

        difference_stage3 = torch.abs(semantic_features_t1[2] - semantic_features_t2[2])
        cues_stage3=torch.cat([semantic_features_t1[2], semantic_features_t2[2], difference_stage3, class_score_difference], dim=1)
        aggregated_stage3=self.scale_aggregation3(cues_stage3)
        aggregated_stage3=F.interpolate(aggregated_stage3, size=(H//16, W//16), mode='bilinear')
        
        semantic_t1_stage2 = F.interpolate(semantic_features_t1[1], size=(H//8, W//8), mode='bilinear')
        semantic_t2_stage2 = F.interpolate(semantic_features_t2[1], size=(H//8, W//8), mode='bilinear')
        difference_stage2 = torch.abs(semantic_t1_stage2 - semantic_t2_stage2)
        score_difference_stage2 = F.interpolate(class_score_difference, size=(H//8, W//8), mode='bilinear')
        cues_stage2 = torch.cat([semantic_t1_stage2, semantic_t2_stage2, difference_stage2, score_difference_stage2, spatial_t1_h8, spatial_t2_h8], dim=1)
        aggregated_stage2 = self.scale_aggregation2(cues_stage2)
        
        semantic_t1_stage1 = F.interpolate(semantic_features_t1[0], size=(H//4, W//4), mode='bilinear')
        semantic_t2_stage1 = F.interpolate(semantic_features_t2[0], size=(H//4, W//4), mode='bilinear')
        difference_stage1 = torch.abs(semantic_t1_stage1 - semantic_t2_stage1)
        score_difference_stage1 = F.interpolate(class_score_difference, size=(H//4, W//4), mode='bilinear')
        cues_stage1 = torch.cat([semantic_t1_stage1, semantic_t2_stage1, difference_stage1, score_difference_stage1, spatial_t1_h4, spatial_t2_h4], dim=1)
        aggregated_stage1 = self.scale_aggregation1(cues_stage1)
        aggregated_stage1 = F.interpolate(aggregated_stage1, size=(H//4, W//4), mode='bilinear')

        decoded_feature=self.decoder_block3(aggregated_stage4, aggregated_stage3)
        decoded_feature=self.decoder_block2(decoded_feature, aggregated_stage2)
        decoded_feature=self.decoder_block1(decoded_feature, aggregated_stage1)
        
        target_h, target_w = decoded_feature.shape[-2], decoded_feature.shape[-1]
        decoded_scores_t1 = F.interpolate(class_scores_t1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        decoded_scores_t2 = F.interpolate(class_scores_t2, size=(target_h, target_w), mode='bilinear', align_corners=False)
        decoded_score_difference = torch.abs(F.softmax(decoded_scores_t1, dim=1) - F.softmax(decoded_scores_t2, dim=1))

        refined_feature = self.dual_axis_refinement(
            decoded_feature,
            decoded_scores_t1,
            decoded_scores_t2,
            decoded_score_difference,
        )
        decoded_feature = decoded_feature + refined_feature
        
        mask_logits=self.binary_change_head(decoded_feature)
        mask_logits=F.interpolate(mask_logits, size=(H, W), mode='bilinear')
        return mask_logits,class_scores_t1_full,class_scores_t2_full,category_aware_t1,category_aware_t2

