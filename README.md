# Learning Local-Structure-Aware Text-Visual Alignment for Remote Sensing SCD

## Setup

```bash
pip install -r requirements.txt
```

Download the SAM 3 checkpoint and place it at `model_weights/sam3.pt`, or pass
its path with `--sam3_path`.

Prepare SECOND and Landsat-SCD with the directory names used in
`configs/SECOND.json` and `configs/Landsat-SCD.json`. Paired images and labels
must share the same filename.

## Train

```bash
python train.py --config configs/SECOND.json \
  --sam3_path model_weights/sam3.pt \
  --save_path checkpoints/second
```

Use `configs/Landsat-SCD.json` for Landsat-SCD.

## Evaluate

```bash
python evaluate.py --config configs/SECOND.json \
  --checkpoint checkpoints/second/best_model.pth
```

The script reports OA, mIoU, Fscd, and SeK.

## Save predictions

```bash
python predict.py --config configs/SECOND.json \
  --checkpoint checkpoints/second/best_model.pth \
  --output_dir results/second
```

Prediction masks are written to `change_mask/`, `t1_scd_mask/`, and
`t2_scd_mask/`.
