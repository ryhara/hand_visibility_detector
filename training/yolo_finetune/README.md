# YOLO hand-detector fine-tuning on HInt

Fine-tunes WiLoR-mini's `detector.pt` (Ultralytics YOLO, 2 classes: `left`/`right`) on the HInt partial-annotation dataset, excluding ego4d (frames not provided).

Class indices match the WiLoR pipeline:
- `0 = left`, `1 = right`

## Data

Source: `/path/to/HInt_annotation_partial/`

- `TRAIN_epick_img` + `TRAIN_newdays_img` → `train`
- `VAL_epick_img`   + `VAL_newdays_img`   → `val`
- `TEST_epick_img`  + `TEST_newdays_img`  → `test`

Each JSON describes one hand: `[{"bbox": [[x1,y1,x2,y2]], ...}]`. Handedness is read from the `_r` / `_l` filename suffix.

> HInt labels only one hand per frame; another visible hand will look like background to YOLO. This is a known dataset limitation.

## Usage

1. Convert HInt to YOLO layout (symlinks the jpgs; no copy):

```bash
uv run python training/yolo_finetune/prepare_dataset.py
# -> training/yolo_finetune/hint_yolo/{images,labels}/{train,val,test}, data.yaml
```

2. Fine-tune from `detector.pt` (downloaded from `warmshao/WiLoR-mini` on first run):

```bash
uv run python training/yolo_finetune/train.py \
    --epochs 50 --batch 16 --imgsz 640 --device 0
```

Outputs:
- `runs/hint_yolo/finetune/weights/best.pt` — fine-tuned checkpoint
- `runs/hint_yolo/{baseline,finetuned}_<name>_test/` — test-split eval dirs
- Side-by-side metrics table with Δ printed at the end (see below)

To use the fine-tuned weights with WiLoR-mini, place `best.pt` somewhere and pass `wilor_pretrained_dir` so that `pretrained_models/detector.pt` resolves to it, or load it directly via `ultralytics.YOLO(best.pt)`.

3. (Optional) Re-run the baseline vs. fine-tuned comparison without retraining:

```bash
uv run python training/yolo_finetune/evaluate.py \
    --weights baseline=auto finetuned=runs/hint_yolo/finetune/weights/best.pt
```

`baseline=auto` downloads WiLoR's `detector.pt` from `warmshao/WiLoR-mini` (same checkpoint as the pipeline uses). You can add more entries, e.g. `--weights baseline=auto ftA=runA/best.pt ftB=runB/best.pt`; when exactly 2 entries are given, a Δ column (`entry2 - entry1`) is shown.

Positive Δ is green, negative is red (when stdout is a TTY).

## Hyperparameter alignment with the WiLoR paper

The WiLoR paper (arxiv:2409.12259, §5.1) trained the detector **from scratch** on WHIM (~2M imgs) for 200 epochs on 2×RTX 4090 over ~3 weeks. We fine-tune on the much smaller HInt subset, so some settings are intentionally relaxed.

| Setting | Paper (scratch) | This script (FT) | Reason |
|---|---|---|---|
| optimizer | Adam | Adam | matched |
| box / cls / dfl weights | 15 / 0.5 / 1.5 | 15 / 0.5 / 1.5 | matched (paper §5.1: λ₂/λ₀/λ₁) |
| rotation | ±60° | ±60° | matched |
| scale jitter | [0.5, 1] | `scale=0.5` | matched (Ultralytics convention is ±scale) |
| mosaic | 0.7 | 0.7 | matched |
| mixup | "used" (no value) | 0.1 | matched (low default) |
| random masking | yes | `erasing=0.4` | matched (Ultralytics' erasing covers it) |
| **fliplr** | — | **0.0** | **MUST be 0**: left/right are separate classes; Ultralytics' hflip does NOT swap class IDs |
| lr0 | 0.01 | 1e-3 | relaxed (avoid forgetting WHIM features) |
| lrf (final/initial) | 1e-6 / 0.01 ≈ 1e-4 | 0.01 | relaxed |
| epochs | 200 | 50 | relaxed |
| patience | 30 | 15 | relaxed |
| batch | 256 (2×4090) | 16 | hardware dependent |
| imgsz | not specified | 640 | Ultralytics default |

All settings are CLI args so you can override any of them, e.g. `--epochs 100 --lr0 5e-4 --batch 32`.
