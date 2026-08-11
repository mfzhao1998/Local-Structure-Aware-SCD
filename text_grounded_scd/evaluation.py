import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import cal_kappa, get_hist


def predict_probabilities(model, t1, t2):
    transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, 3, [2, 3]), [3]),
    ]
    inverse_transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, -1, [2, 3]),
        lambda x: torch.rot90(x, -2, [2, 3]),
        lambda x: torch.rot90(x, -3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(torch.flip(x, [3]), -1, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), -3, [2, 3]),
    ]

    change_logits = []
    semantic_logits_t1 = []
    semantic_logits_t2 = []
    for transform, inverse_transform in zip(transforms, inverse_transforms):
        change, semantic_t1, semantic_t2, _, _ = model(
            transform(t1), transform(t2)
        )
        change_logits.append(inverse_transform(change))
        semantic_logits_t1.append(inverse_transform(semantic_t1))
        semantic_logits_t2.append(inverse_transform(semantic_t2))

    return (
        torch.sigmoid(torch.stack(change_logits).mean(dim=0)),
        F.softmax(torch.stack(semantic_logits_t1).mean(dim=0), dim=1),
        F.softmax(torch.stack(semantic_logits_t2).mean(dim=0), dim=1),
    )


def generate_scd_predictions(
    change_probability,
    semantic_probability_t1,
    semantic_probability_t2,
    threshold,
):
    predicted_change = (change_probability.squeeze(1) > threshold).long()
    semantic_t1 = torch.argmax(semantic_probability_t1, dim=1)
    semantic_t2 = torch.argmax(semantic_probability_t2, dim=1)
    conflict = (
        (predicted_change == 1)
        & (semantic_t1 == semantic_t2)
    )

    if conflict.any():
        confidence_t1 = torch.max(semantic_probability_t1, dim=1).values
        confidence_t2 = torch.max(semantic_probability_t2, dim=1).values
        second_t1 = torch.topk(
            semantic_probability_t1, 2, dim=1
        ).indices[:, 1]
        second_t2 = torch.topk(
            semantic_probability_t2, 2, dim=1
        ).indices[:, 1]
        weaker_t1 = conflict & (confidence_t1 < confidence_t2)
        weaker_t2 = conflict & (confidence_t1 >= confidence_t2)
        semantic_t1[weaker_t1] = second_t1[weaker_t1]
        semantic_t2[weaker_t2] = second_t2[weaker_t2]

    return (
        predicted_change,
        semantic_t1 * predicted_change,
        semantic_t2 * predicted_change,
    )


def evaluate_model(
    model,
    val_loader,
    device,
    config,
    thresholds=None,
    print_metrics=True,
    show_progress=True,
):
    model.eval()
    num_classes_sem = config["num_classes"]
    if thresholds is None:
        thresholds = [0.5]

    hist_cd = {threshold: np.zeros((2, 2)) for threshold in thresholds}
    hist_sem_global = {
        threshold: np.zeros((num_classes_sem, num_classes_sem))
        for threshold in thresholds
    }
    hist_sem_masked = {
        threshold: np.zeros((num_classes_sem, num_classes_sem))
        for threshold in thresholds
    }

    def update_hist(prediction, target, num_classes, histogram):
        valid = (target >= 0) & (target < num_classes)
        histogram += get_hist(
            prediction[valid], target[valid], num_classes
        )

    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc="Validating",
            leave=False,
            disable=not show_progress,
        ):
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            target_cd = batch["label"].to(device).long()
            target_sem_t1 = batch.get("sem_t1")
            target_sem_t2 = batch.get("sem_t2")
            prob_cd, prob_sem_t1, prob_sem_t2 = predict_probabilities(
                model, t1, t2
            )

            target_cd_np = target_cd.squeeze(1).cpu().numpy()

            if target_sem_t1 is not None and target_sem_t2 is not None:
                target_sem_t1_np = target_sem_t1.squeeze(1).numpy()
                target_sem_t2_np = target_sem_t2.squeeze(1).numpy()
                changed_target = target_cd_np == 1

            for threshold in thresholds:
                predicted_change, final_sem_t1, final_sem_t2 = (
                    generate_scd_predictions(
                        prob_cd,
                        prob_sem_t1,
                        prob_sem_t2,
                        threshold,
                    )
                )
                pred_cd = predicted_change.cpu().numpy().astype(np.uint8)
                update_hist(pred_cd, target_cd_np, 2, hist_cd[threshold])

                if target_sem_t1 is None or target_sem_t2 is None:
                    continue

                final_sem_t1 = final_sem_t1.cpu().numpy()
                final_sem_t2 = final_sem_t2.cpu().numpy()
                update_hist(
                    final_sem_t1,
                    target_sem_t1_np,
                    num_classes_sem,
                    hist_sem_global[threshold],
                )
                update_hist(
                    final_sem_t2,
                    target_sem_t2_np,
                    num_classes_sem,
                    hist_sem_global[threshold],
                )

                masked_target_t1 = target_sem_t1_np.copy()
                masked_target_t2 = target_sem_t2_np.copy()
                masked_target_t1[~changed_target] = 255
                masked_target_t2[~changed_target] = 255
                update_hist(
                    final_sem_t1,
                    masked_target_t1,
                    num_classes_sem,
                    hist_sem_masked[threshold],
                )
                update_hist(
                    final_sem_t2,
                    masked_target_t2,
                    num_classes_sem,
                    hist_sem_masked[threshold],
                )

    best_miou_cd = -1.0
    best_threshold = thresholds[0]
    final_hist_cd = hist_cd[best_threshold]
    final_hist_sem_global = hist_sem_global[best_threshold]
    final_hist_sem_masked = hist_sem_masked[best_threshold]

    for threshold in thresholds:
        current_hist = hist_cd[threshold]
        iou_cd = np.diag(current_hist) / (
            current_hist.sum(1)
            + current_hist.sum(0)
            - np.diag(current_hist)
            + 1e-10
        )
        miou_cd = np.mean(iou_cd)
        if miou_cd > best_miou_cd:
            best_miou_cd = miou_cd
            best_threshold = threshold
            final_hist_cd = current_hist
            final_hist_sem_global = hist_sem_global[threshold]
            final_hist_sem_masked = hist_sem_masked[threshold]

    iou_cd = np.diag(final_hist_cd) / (
        final_hist_cd.sum(1)
        + final_hist_cd.sum(0)
        - np.diag(final_hist_cd)
        + 1e-10
    )
    iou_sem = np.diag(final_hist_sem_masked) / (
        final_hist_sem_masked.sum(1)
        + final_hist_sem_masked.sum(0)
        - np.diag(final_hist_sem_masked)
        + 1e-10
    )
    f1_sem = 2 * np.diag(final_hist_sem_masked) / (
        final_hist_sem_masked.sum(1)
        + final_hist_sem_masked.sum(0)
        + 1e-10
    )
    hist_no_unchanged = final_hist_sem_global.copy()
    hist_no_unchanged[0, 0] = 0
    kappa = cal_kappa(hist_no_unchanged)

    results = {
        "OA": np.diag(final_hist_cd).sum() / (final_hist_cd.sum() + 1e-10),
        "mIoU_CD": np.mean(iou_cd),
        "IoU_Changed": iou_cd[1],
        "mIoU_Semantic": np.nanmean(iou_sem[1:]),
        "fSCD": np.nanmean(f1_sem[1:]),
        "SeK": kappa * math.exp(iou_cd[1] - 1),
        "kappa": kappa,
        "Best_Th": best_threshold,
    }
    if print_metrics:
        print(
            f"OA: {results['OA']:.4f}, "
            f"mIoU: {results['mIoU_CD']:.4f}, "
            f"Fscd: {results['fSCD']:.4f}, "
            f"Sek: {results['SeK']:.4f}"
        )
    return results
