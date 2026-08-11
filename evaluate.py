import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from text_grounded_scd.evaluation import evaluate_model
from text_grounded_scd.model import TextGroundedSCD


def build_image_transform(config):
    return transforms.Compose(
        [
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-grounded semantic change detection evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/SECOND.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (overrides config)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold; otherwise select 0.4/0.5/0.6 on validation",
    )
    return parser.parse_args()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        t1_dir,
        t2_dir,
        label_dir,
        sem_t1_dir=None,
        sem_t2_dir=None,
        img_size=512,
        config=None,
    ):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.label_dir = label_dir
        self.sem_t1_dir = sem_t1_dir
        self.sem_t2_dir = sem_t2_dir
        self.img_size = img_size
        self.img_list = sorted(
            filename
            for filename in os.listdir(t1_dir)
            if filename.lower().endswith((".png", ".jpg", ".tif", ".tiff"))
        )
        self.transform = build_image_transform(config)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_name = self.img_list[index]
        t1 = self.transform(
            Image.open(os.path.join(self.t1_dir, image_name)).convert("RGB")
        )
        t2 = self.transform(
            Image.open(os.path.join(self.t2_dir, image_name)).convert("RGB")
        )

        label = None
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, image_name)
            if os.path.exists(label_path):
                label_image = Image.open(label_path).convert("L")
                label_image = label_image.resize(
                    (self.img_size, self.img_size), resample=Image.NEAREST
                )
                label = torch.from_numpy(np.array(label_image)).long()
                label = (label > 127).long().unsqueeze(0)

        semantic_t1 = self._load_semantic_label(self.sem_t1_dir, image_name)
        semantic_t2 = self._load_semantic_label(self.sem_t2_dir, image_name)
        if label is None and semantic_t1 is not None and semantic_t2 is not None:
            label = (semantic_t1 != semantic_t2).long()

        sample = {"t1": t1, "t2": t2, "label": label, "name": image_name}
        if semantic_t1 is not None:
            sample["sem_t1"] = semantic_t1
        if semantic_t2 is not None:
            sample["sem_t2"] = semantic_t2
        return sample

    def _load_semantic_label(self, directory, image_name):
        if not directory:
            return None
        path = os.path.join(directory, image_name)
        if not os.path.exists(path):
            return None
        image = Image.open(path).convert("L")
        image = image.resize(
            (self.img_size, self.img_size), resample=Image.NEAREST
        )
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


def load_model(config, device, checkpoint_override=None):
    print("Loading model architecture...")
    model = TextGroundedSCD(
        checkpoint_path=config.get("sam3_path"),
        img_size=config["img_size"],
        num_classes=config["num_classes"],
        class_names=config.get("class_names"),
    )

    checkpoint_path = checkpoint_override or config.get("best_model_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get(
        "state_dict", checkpoint.get("model", checkpoint)
    )
    state_dict = {
        key.replace("module.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully.")

    model.to(device)
    model.eval()
    return model


def get_prediction_split(config):
    paths = config["paths"]
    if paths.get("test_t1") and paths.get("test_t2"):
        return "test"
    return "val"


def build_dataset(config, split):
    paths = config["paths"]
    return CustomDataset(
        t1_dir=paths[f"{split}_t1"],
        t2_dir=paths[f"{split}_t2"],
        label_dir=paths.get(f"{split}_mask"),
        sem_t1_dir=paths.get(f"{split}_sem_t1"),
        sem_t2_dir=paths.get(f"{split}_sem_t2"),
        img_size=config["img_size"],
        config=config,
    )


def select_inference_threshold(model, config, device, threshold=None):
    if threshold is not None:
        return threshold

    if get_prediction_split(config) == "val":
        return 0.5

    validation_dataset = build_dataset(config, "val")
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config.get("num_workers", 8),
    )
    results = evaluate_model(
        model,
        validation_loader,
        device,
        config,
        thresholds=[0.4, 0.5, 0.6],
        print_metrics=False,
        show_progress=False,
    )
    return results["Best_Th"]


def run_evaluation():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    print(f"Evaluation Configuration: {args.config}")
    model = load_model(config, device, checkpoint_override=args.checkpoint)
    threshold = select_inference_threshold(
        model,
        config,
        device,
        threshold=args.threshold,
    )
    prediction_split = get_prediction_split(config)
    test_dataset = build_dataset(config, prediction_split)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=config.get("num_workers", 8),
    )

    print(f"Starting evaluation on {len(test_dataset)} images...")
    evaluate_model(
        model,
        test_loader,
        device,
        config,
        thresholds=[threshold],
    )


if __name__ == "__main__":
    run_evaluation()
