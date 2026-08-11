import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from text_grounded_scd.data import get_loaders_from_config
from text_grounded_scd.evaluation import evaluate_model
from text_grounded_scd.model import TextGroundedSCD
from text_grounded_scd.options import TrainOptions


class BinaryChangeLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=2.0):
        super(BinaryChangeLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        if self.bce_loss.pos_weight.device != inputs.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
        loss_bce = self.bce_loss(inputs, targets.float())
        inputs_sigmoid = torch.sigmoid(inputs)
        smooth = 1e-5
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1).float()
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        loss_dice = 1 - dice_score
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class SemanticConsistencyLoss(nn.Module):
    def __init__(self, margin=0.2, prediction_weight=1.0, ohem_ratio=0.2):
        super(SemanticConsistencyLoss, self).__init__()
        self.margin = margin
        self.prediction_weight = prediction_weight
        self.ratio = ohem_ratio

    def forward(self, sem_t1, sem_t2, cls_t1, cls_t2, mask):
        if mask.shape[-2:] != sem_t1.shape[-2:]:
            mask = F.interpolate(mask.float(), size=sem_t1.shape[-2:], mode='nearest')
        if mask.max() > 1:
            mask = mask / 255.0
        sem_t1_norm = F.normalize(sem_t1, dim=1)
        sem_t2_norm = F.normalize(sem_t2, dim=1)
        sim = (sem_t1_norm * sem_t2_norm).sum(dim=1, keepdim=True)
        loss_unchanged = (1 - sim) * (1 - mask)
        loss_changed = F.relu(sim - (1 - self.margin)) * mask
        num_unchanged = (1 - mask).sum().clamp(min=1.0)
        num_changed = mask.sum().clamp(min=1.0)
        loss_feature = loss_unchanged.sum() / num_unchanged + loss_changed.sum() / num_changed
        p1 = F.softmax(cls_t1, dim=1)
        p2 = F.softmax(cls_t2, dim=1)
        s_diff = torch.norm(p1 - p2, p=2, dim=1, keepdim=True) / 1.414
        if s_diff.shape[-2:] != mask.shape[-2:]:
            s_diff = F.interpolate(s_diff, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        diff = torch.abs(s_diff - mask)
        diff_flat = diff.view(-1)
        num_hard = int(diff_flat.numel() * self.ratio)
        if num_hard > 0:
            hard_prediction_errors, _ = torch.topk(diff_flat, num_hard)
            loss_prediction = hard_prediction_errors.mean()
        else:
            loss_prediction = diff_flat.mean()
        loss_consistency = loss_feature + self.prediction_weight * loss_prediction
        return loss_consistency, loss_feature, loss_prediction

def semantic_focal_loss(inputs, targets, alpha=None, gamma=2, ignore_index=255, reduction='mean'):
    ce_loss = F.cross_entropy(inputs, targets, weight=alpha, ignore_index=ignore_index, reduction='none')
    logpt = -ce_loss
    pt = torch.exp(logpt)
    focal_term = (1 - pt) ** gamma
    loss = focal_term * ce_loss
    if reduction == 'mean':
        valid_mask = (targets != ignore_index).float()
        return loss.sum() / (valid_mask.sum() + 1e-6)
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def main():
    opt = TrainOptions()
    args = opt.parse()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    train_loader, val_loader = get_loaders_from_config(args.config)
    img_size = cfg['img_size']
    num_classes = cfg.get('num_classes')
    class_names = cfg.get('class_names') 
    device = torch.device("cuda")
    model = TextGroundedSCD(
        checkpoint_path=args.sam3_path, 
        img_size=img_size, 
        num_classes=num_classes,
        class_names=class_names)
    model.to(device)
    backbone_params = []
    head_params = []
    head_keywords = [
        'binary_change_head', 'dual_axis_refinement',
        'decoder_block1', 'decoder_block2', 'decoder_block3',
        'scale_aggregation1', 'scale_aggregation2',
        'scale_aggregation3', 'scale_aggregation4',
        'dynamic_local_alignment', 'global_semantic_injection',
        'cnn_spatial_branch'
    ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)
    params_group = [
        {"params": backbone_params, "lr": args.base_lr},
        {"params": head_params, "lr": args.base_lr * args.head_lr_mult}, 
    ]
    optim = torch.optim.AdamW(params_group, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=10, min_lr=1e-6)
    criterion_cd = BinaryChangeLoss(pos_weight=2.0).to(device)
    criterion_sc = SemanticConsistencyLoss(
        margin=0.2, prediction_weight=1.0
    ).to(device)
    os.makedirs(args.save_path, exist_ok=True)
    best_score = 0.0
    class_weights = None
    if 'class_weights' in cfg:
        class_weights = torch.tensor(cfg['class_weights']).float().to(device)
    for epoch in range(args.epoch):
        model.train()
        metrics = {"total": 0.0, "cd": 0.0, "sem": 0.0, "sc": 0.0}
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
        for i, batch in enumerate(train_bar):
            t1 = batch['t1'].to(device) 
            t2 = batch['t2'].to(device)
            target = batch['label'].to(device)
            has_sem = 'sem_t1' in batch and 'sem_t2' in batch
            if has_sem:
                label_sem_t1 = batch['sem_t1'].to(device)
                label_sem_t2 = batch['sem_t2'].to(device)
            if torch.rand(1).item() > 0.5:
                t1, t2 = t2, t1
                if has_sem:
                    label_sem_t1, label_sem_t2 = label_sem_t2, label_sem_t1
            optim.zero_grad()
            (
                change_logits,
                class_scores_t1,
                class_scores_t2,
                category_aware_t1,
                category_aware_t2,
            ) = model(t1, t2)
            loss_cd = criterion_cd(change_logits, target)
            total_loss = loss_cd
            loss_sem_val = 0.0
            loss_sc_val = 0.0
            if has_sem:
                loss_sem_t1_focal = semantic_focal_loss(
                    class_scores_t1,
                    label_sem_t1.squeeze(1).long(),
                    alpha=class_weights,
                    reduction='none',
                )
                loss_sem_t2_focal = semantic_focal_loss(
                    class_scores_t2,
                    label_sem_t2.squeeze(1).long(),
                    alpha=class_weights,
                    reduction='none',
                )
                valid_mask = target.float().squeeze(1)
                num_valid = valid_mask.sum() + 1e-6
                loss_sem_t1_focal = (loss_sem_t1_focal * valid_mask).sum() / num_valid
                loss_sem_t2_focal = (loss_sem_t2_focal * valid_mask).sum() / num_valid
                loss_sem = (loss_sem_t1_focal + loss_sem_t2_focal) / 2.0
                loss_sc, _, _ = criterion_sc(
                    category_aware_t1,
                    category_aware_t2,
                    class_scores_t1,
                    class_scores_t2,
                    target,
                )
                total_loss = loss_cd + loss_sem + 0.5 * loss_sc
                loss_sem_val = loss_sem.item()
                loss_sc_val = loss_sc.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            metrics['total'] += total_loss.item()
            metrics['cd'] += loss_cd.item()
            metrics['sem'] += loss_sem_val
            metrics['sc'] += loss_sc_val
            train_bar.set_postfix({
                'Loss(Total)': metrics['total']/(i+1),
                'Loss(CD)': metrics['cd']/(i+1),
                'Loss(Sem)': metrics['sem']/(i+1),
                'Loss(SC)': metrics['sc']/(i+1)})
        steps = len(train_loader)
        print(
            f"Epoch {epoch+1} Summary: "
            f"Loss(Total): {metrics['total']/steps:.4f}, "
            f"Loss(CD): {metrics['cd']/steps:.4f}, "
            f"Loss(Sem): {metrics['sem']/steps:.4f}, "
            f"Loss(SC): {metrics['sc']/steps:.4f}"
        )
        val = evaluate_model(model, val_loader, device, config=cfg)
        if val['mIoU_CD'] > best_score:
            best_score = val['mIoU_CD']
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            print(f"  *** Best Model Saved (mIoU: {best_score * 100:.2f}%) ***")
        scheduler.step(val['mIoU_CD'])


if __name__ == "__main__":
    main()
