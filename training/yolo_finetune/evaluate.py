"""Evaluate one or more YOLO checkpoints on the HInt test split and print a
side-by-side comparison table.

Typical use:

    # baseline (WiLoR's detector.pt from HF) vs. fine-tuned best.pt
    uv run python training/yolo_finetune/evaluate.py \
        --weights baseline=<auto> finetuned=runs/hint_yolo/finetune/weights/best.pt

`<auto>` (or omitting --weights altogether) resolves the WiLoR pretrained
detector via huggingface_hub, exactly the same checkpoint as
`wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline.init_models`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


WILOR_MINI_REPO_ID = "warmshao/WiLoR-mini"

# Colors for terminal diff output; auto-disabled when stdout isn't a tty.
_USE_COLOR = sys.stdout.isatty()
_GREEN = "\033[32m" if _USE_COLOR else ""
_RED = "\033[31m" if _USE_COLOR else ""
_DIM = "\033[2m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


def resolve_baseline(cache_dir: Path) -> Path:
    """Download detector.pt from HF if missing; return its path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / "pretrained_models" / "detector.pt"
    if not local.exists():
        print(f"downloading detector.pt from {WILOR_MINI_REPO_ID} -> {local}")
        hf_hub_download(
            repo_id=WILOR_MINI_REPO_ID,
            subfolder="pretrained_models",
            filename="detector.pt",
            local_dir=str(cache_dir),
        )
    return local


def evaluate_one(
    *,
    label: str,
    weights: Path,
    data: Path,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/hint_yolo",
    name_suffix: str = "eval",
) -> dict[str, float]:
    """Run model.val on `weights` and return a flat dict of metrics."""
    print(f"\n=== evaluating [{label}]: {weights} ===")
    model = YOLO(str(weights))
    m = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=f"{label}_{name_suffix}",
        exist_ok=True,
        verbose=False,
    )
    box = m.box
    out: dict[str, float] = {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "Precision": float(box.mp),
        "Recall": float(box.mr),
    }
    # Per-class metrics (left / right).
    try:
        names = getattr(m, "names", None) or model.names
        for i, idx in enumerate(box.ap_class_index):
            cname = names[int(idx)]
            out[f"AP50[{cname}]"] = float(box.ap50[i])
            # `box.maps` is indexed by class id, not by ap_class_index position.
            out[f"AP50-95[{cname}]"] = float(box.maps[int(idx)])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not extract per-class metrics: {e}")
    return out


def _fmt_delta(d: float) -> str:
    if d > 0:
        return f"{_GREEN}+{d:.4f}{_RESET}"
    if d < 0:
        return f"{_RED}{d:.4f}{_RESET}"
    return f"{_DIM}  0.0000{_RESET}"


def print_comparison(results: dict[str, dict[str, float]]) -> None:
    """results: label -> {metric: value}.  When 2 labels are present, also
    prints a Δ column (label2 - label1)."""
    if not results:
        print("no results to display.")
        return

    labels = list(results.keys())
    # Union of metric keys, in insertion order of the first run.
    metric_keys: list[str] = []
    for label in labels:
        for k in results[label]:
            if k not in metric_keys:
                metric_keys.append(k)

    col_w = max(14, max(len(l) for l in labels) + 2)
    metric_w = max(len(k) for k in metric_keys) + 2

    header = "Metric".ljust(metric_w) + "".join(l.rjust(col_w) for l in labels)
    if len(labels) == 2:
        header += "Δ".rjust(col_w)
    print("\n" + header)
    print("-" * len(header))

    for k in metric_keys:
        row = k.ljust(metric_w)
        vals = [results[l].get(k) for l in labels]
        for v in vals:
            row += ("-".rjust(col_w) if v is None else f"{v:.4f}".rjust(col_w))
        if len(labels) == 2 and vals[0] is not None and vals[1] is not None:
            d = vals[1] - vals[0]
            # _fmt_delta already pads via ANSI; right-align by length sans color.
            txt = _fmt_delta(d)
            visible = txt
            for c in (_GREEN, _RED, _DIM, _RESET):
                visible = visible.replace(c, "")
            pad = col_w - len(visible)
            row += (" " * max(pad, 0)) + txt
        print(row)
    print()


def parse_weights_arg(items: list[str], cache_dir: Path) -> list[tuple[str, Path]]:
    """Parse `--weights LABEL=PATH ...`.  PATH==`auto` (or omitted on the
    baseline default) resolves the WiLoR detector from HuggingFace."""
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--weights expects LABEL=PATH, got: {item!r}")
        label, path = item.split("=", 1)
        if path.lower() in ("auto", "hf", "wilor"):
            p = resolve_baseline(cache_dir)
        else:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(p)
        out.append((label, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "hint_yolo" / "data.yaml",
    )
    ap.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help="LABEL=PATH pairs, e.g. baseline=auto finetuned=runs/.../best.pt. "
             "Use 'auto' as PATH to fetch WiLoR's detector.pt from HuggingFace.",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "wilor_cache",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "hint_yolo"),
    )
    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found. Run prepare_dataset.py first."
        )

    # Default: baseline vs. expected best.pt from the default training run.
    if not args.weights:
        default_best = (
            Path(__file__).resolve().parent
            / "runs" / "hint_yolo" / "finetune" / "weights" / "best.pt"
        )
        items = ["baseline=auto"]
        if default_best.exists():
            items.append(f"finetuned={default_best}")
        else:
            print(f"[info] no fine-tuned best.pt at {default_best}; "
                  "evaluating baseline only. Pass --weights LABEL=PATH ... for more.")
        args.weights = items

    pairs = parse_weights_arg(args.weights, args.cache_dir)

    results: dict[str, dict[str, float]] = {}
    for label, w in pairs:
        results[label] = evaluate_one(
            label=label,
            weights=w,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
        )

    print_comparison(results)


if __name__ == "__main__":
    main()
