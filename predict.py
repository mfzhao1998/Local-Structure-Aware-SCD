import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from text_grounded_scd.evaluation import (
    generate_scd_predictions,
    predict_probabilities,
)
from evaluate import (
    build_image_transform,
    get_prediction_split,
    load_model,
    select_inference_threshold,
)


SECOND_COLORS = np.array(
    [
        [255, 255, 255],
        [0, 0, 255],
        [128, 128, 128],
        [0, 255, 0],
        [0, 100, 0],
        [255, 0, 0],
        [255, 255, 0],
    ],
    dtype=np.uint8,
)

LANDSAT_SCD_COLORS = np.array(
    [
        [255, 255, 255],
        [0, 155, 0],
        [255, 165, 0],
        [230, 30, 100],
        [0, 170, 240],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save text-grounded SCD change and semantic masks"
    )
    parser.add_argument("--config", default="configs/SECOND.json")
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold; otherwise use inference threshold selection",
    )
    return parser.parse_args()


def get_color_palette(dataset_name):
    if "landsat" in dataset_name.lower():
        return LANDSAT_SCD_COLORS
    return SECOND_COLORS


def colorize(label, num_classes, palette):
    if num_classes > len(palette):
        raise ValueError(
            f"Color palette has {len(palette)} entries, "
            f"but the model has {num_classes} classes."
        )
    return palette[label]


def run():
    args = parse_args()
    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        config,
        device,
        checkpoint_override=args.checkpoint,
    )
    threshold = select_inference_threshold(
        model,
        config,
        device,
        threshold=args.threshold,
    )
    color_palette = get_color_palette(config["dataset_name"])

    paths = config["paths"]
    prediction_split = get_prediction_split(config)
    t1_dir = paths[f"{prediction_split}_t1"]
    t2_dir = paths[f"{prediction_split}_t2"]
    output_root = args.output_dir or f"results_{config['dataset_name']}"
    output_dirs = {
        "change": os.path.join(output_root, "change_mask"),
        "t1_scd": os.path.join(output_root, "t1_scd_mask"),
        "t2_scd": os.path.join(output_root, "t2_scd_mask"),
    }
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)

    transform = build_image_transform(config)
    image_names = sorted(
        name
        for name in os.listdir(t1_dir)
        if name.lower().endswith((".jpg", ".png", ".tif", ".tiff"))
    )

    with torch.no_grad():
        for image_name in tqdm(image_names, desc="Saving masks"):
            t2_path = os.path.join(t2_dir, image_name)
            if not os.path.exists(t2_path):
                continue

            t1 = transform(
                Image.open(os.path.join(t1_dir, image_name)).convert("RGB")
            ).unsqueeze(0).to(device)
            t2 = transform(
                Image.open(t2_path).convert("RGB")
            ).unsqueeze(0).to(device)

            (
                change_probability,
                semantic_probability_t1,
                semantic_probability_t2,
            ) = predict_probabilities(model, t1, t2)
            predicted_change, semantic_t1, semantic_t2 = (
                generate_scd_predictions(
                    change_probability,
                    semantic_probability_t1,
                    semantic_probability_t2,
                    threshold,
                )
            )
            predicted_change = (
                predicted_change.squeeze(0).cpu().numpy().astype(np.uint8)
            )
            semantic_t1 = semantic_t1.squeeze(0).cpu().numpy()
            semantic_t2 = semantic_t2.squeeze(0).cpu().numpy()

            output_name = f"{os.path.splitext(image_name)[0]}.png"
            Image.fromarray(predicted_change * 255).save(
                os.path.join(output_dirs["change"], output_name)
            )
            Image.fromarray(
                colorize(
                    semantic_t1,
                    config["num_classes"],
                    color_palette,
                )
            ).save(os.path.join(output_dirs["t1_scd"], output_name))
            Image.fromarray(
                colorize(
                    semantic_t2,
                    config["num_classes"],
                    color_palette,
                )
            ).save(os.path.join(output_dirs["t2_scd"], output_name))


if __name__ == "__main__":
    run()
