"""Evaluate one or more trained HandVisibilityNet checkpoints on a val set.

For every checkpoint listed in the config's ``models`` section this writes,
under its own subdirectory ``output.dir/<name>/``:

    * ``<name>.csv``             -- per-subset + overall metrics
    * ``<name>_pr_curves.jpg``   -- micro-averaged PR curve(s)
    * ``<name>_<image_id>.jpg``  -- up to ``n_samples`` GT-vs-Pred panels

It also writes, directly under ``output.dir``, a combined ``comparison.csv``
(overall metrics, one row per model) and ``comparison_pr_curves.jpg`` (all
models' overall PR curves overlaid) so the checkpoints can be compared
directly.

The model architecture (hidden_dim / head-only) is restored from each
checkpoint's own saved ``config`` -- only the checkpoint path and a name are
needed in the eval config.

Usage:
    python -m training.evaluate --config training/configs/hint_eval.yaml

Override any config field via dotted CLI args (OmegaConf style), e.g.
    python -m training.evaluate --config training/configs/hint_eval.yaml \
        output.dir=outputs/eval_run2 batch_size=128
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import HIntHandDataset, COCOWholeBodyHandDataset
from .model import build_model
from .train import _denorm_image, _gt_vs_pred_panel


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Map config metric names -> internal computation. Order here defines CSV
# column order.
ALL_METRIC_TYPES = [
    "accuracy",
    "mAP",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
]


def compute_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    types: list[str],
) -> dict[str, float]:
    """Micro-averaged metrics over all valid ``(sample, keypoint)`` pairs.

    ``logits`` / ``target`` / ``mask`` are ``(N, 21)`` arrays. Only entries
    with ``mask > 0.5`` count. Metrics whose curve is undefined (e.g. a single
    class present) are reported as ``nan``.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    valid = mask > 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    p = probs[valid]
    t = target[valid].astype(np.int32)
    pred = (p >= 0.5).astype(np.int32)

    out: dict[str, float] = {}
    tp = int(((pred == 1) & (t == 1)).sum())
    fp = int(((pred == 1) & (t == 0)).sum())
    fn = int(((pred == 0) & (t == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    def _map() -> float:
        """Macro mAP: per-keypoint average precision, averaged over keypoints
        that have both classes present among their valid entries."""
        aps = []
        for k in range(target.shape[1]):
            vk = mask[:, k] > 0.5
            if vk.sum() < 2:
                continue
            t_k = target[vk, k]
            if t_k.sum() == 0 or t_k.sum() == t_k.shape[0]:
                continue
            aps.append(average_precision_score(t_k, probs[vk, k]))
        return float(np.mean(aps)) if aps else float("nan")

    single_class = t.sum() == 0 or t.sum() == t.shape[0]
    for name in types:
        if name == "accuracy":
            out[name] = float((pred == t).mean()) if t.size else float("nan")
        elif name == "mAP":
            out[name] = _map()
        elif name == "precision":
            out[name] = float(precision)
        elif name == "recall":
            out[name] = float(recall)
        elif name == "f1":
            out[name] = float(
                2 * precision * recall / max(precision + recall, 1e-9)
            )
        elif name == "roc_auc":
            out[name] = (
                float(roc_auc_score(t, p)) if not single_class else float("nan")
            )
        elif name == "pr_auc":
            out[name] = (
                float(average_precision_score(t, p))
                if not single_class
                else float("nan")
            )
        else:
            raise ValueError(f"unknown metric type: {name}")
    return out


def compute_pr_curve(
    logits: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_points: int = 300,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Micro-averaged ``(recall, precision)`` over all valid pairs, or
    ``None`` if undefined."""
    from sklearn.metrics import precision_recall_curve

    valid = mask > 0.5
    if valid.sum() < 2:
        return None
    probs = (1.0 / (1.0 + np.exp(-logits)))[valid]
    t = target[valid]
    if t.sum() == 0 or t.sum() == t.shape[0]:
        return None
    precision, recall, _ = precision_recall_curve(t, probs)
    if len(precision) > max_points:
        idx = np.linspace(0, len(precision) - 1, max_points).astype(int)
        precision, recall = precision[idx], recall[idx]
    return recall, precision


def compute_roc_curve(
    logits: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_points: int = 300,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Micro-averaged ``(fpr, tpr)`` over all valid pairs, or ``None`` if
    undefined (fewer than 2 valid pairs or only one class present)."""
    from sklearn.metrics import roc_curve

    valid = mask > 0.5
    if valid.sum() < 2:
        return None
    probs = (1.0 / (1.0 + np.exp(-logits)))[valid]
    t = target[valid]
    if t.sum() == 0 or t.sum() == t.shape[0]:
        return None
    fpr, tpr, _ = roc_curve(t, probs)
    if len(fpr) > max_points:
        idx = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr, tpr


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Rebuild a model from a checkpoint, using the architecture stored in the
    checkpoint's own ``config``."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    mcfg = ckpt.get("config", {}).get("model", {})
    head_only = bool(ckpt.get("head_only", False))
    model = build_model(
        backbone=str(mcfg.get("backbone", "wilor")),
        # Head-only checkpoints rely on the backbone's own pre-trained weights;
        # full checkpoints overwrite them anyway.
        pretrained=head_only,
        dropout=float(mcfg.get("dropout", 0.0)),
        hidden_dim=int(mcfg.get("hidden_dim", 256)),
        freeze_backbone=bool(mcfg.get("freeze_backbone", head_only)),
    ).to(device)
    if head_only:
        model.load_head_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_val_dataset(cfg, subsets: list[str]):
    dataset_type = str(cfg.data.get("type", "hint")).lower()
    if dataset_type == "hint":
        return HIntHandDataset(
            data_root=cfg.data.hint_root,
            subsets=subsets,
            crop_size=int(cfg.data.crop_size),
            bbox_expand=float(cfg.data.bbox_expand),
            min_bbox_side=float(cfg.data.min_bbox_side),
            train=False,
        )
    if dataset_type == "cocowholebody":
        return COCOWholeBodyHandDataset(
            ann_path=cfg.data.val_ann,
            img_dir=cfg.data.val_img_dir,
            crop_size=int(cfg.data.crop_size),
            bbox_expand=float(cfg.data.bbox_expand),
            min_bbox_side=float(cfg.data.min_bbox_side),
            train=False,
        )
    raise ValueError(f"unknown data.type: {dataset_type}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_vis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, int]]]:
    """Returns ``(logits, target, mask, vis_panels)`` over the whole loader.

    ``vis_panels`` holds up to ``n_vis`` ``(panel_rgb, image_id)`` GT-vs-Pred
    panels from the first batches."""
    all_logits, all_target, all_mask = [], [], []
    vis_panels: list[tuple[np.ndarray, int]] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        logits = model(img)
        logits_np = logits.detach().float().cpu().numpy()
        all_logits.append(logits_np)
        all_target.append(batch["target"].numpy())
        all_mask.append(batch["mask"].numpy())

        if n_vis > 0 and len(vis_panels) < n_vis and "kpts_crop" in batch:
            probs = 1.0 / (1.0 + np.exp(-logits_np))
            kpts_b = batch["kpts_crop"].numpy()
            tgt_b = batch["target"].numpy()
            mask_b = batch["mask"].numpy()
            img_b = batch["image"]
            ids_b = batch["image_id"].numpy()
            for i in range(img_b.shape[0]):
                if len(vis_panels) >= n_vis:
                    break
                # Skip images without any labeled visibility/hand keypoints:
                # nothing would be drawn, so don't visualize or count them.
                if mask_b[i].max() < 0.5:
                    continue
                rgb = _denorm_image(img_b[i])
                panel = _gt_vs_pred_panel(
                    rgb, kpts_b[i], tgt_b[i], probs[i], mask_b[i]
                )
                vis_panels.append((panel, int(ids_b[i])))

    return (
        np.concatenate(all_logits),
        np.concatenate(all_target),
        np.concatenate(all_mask),
        vis_panels,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_metrics_csv(
    path: str,
    metric_types: list[str],
    rows: list[dict],
) -> None:
    """``rows`` are dicts with keys ``subset``, ``n_samples`` and each metric."""
    fieldnames = ["subset", "n_samples"] + metric_types
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_pr_curves(
    path: str,
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
) -> None:
    """``curves`` are ``(label, recall, precision)`` triples."""
    plt.figure(figsize=(6, 6))
    for label, recall, precision in curves:
        plt.plot(recall, precision, label=label, linewidth=1.5)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_roc_curves(
    path: str,
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
) -> None:
    """``curves`` are ``(label, fpr, tpr)`` triples."""
    plt.figure(figsize=(6, 6))
    for label, fpr, tpr in curves:
        plt.plot(fpr, tpr, label=label, linewidth=1.5)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_vis_panels(
    out_dir: str, name: str, panels: list[tuple[np.ndarray, str, int]]
) -> None:
    for panel, subset, image_id in panels:
        bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(out_dir, f"{name}_{subset}_{image_id}.jpg"), bgr
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate_model(cfg, entry, device: torch.device, out_dir: str) -> dict:
    """Evaluate one checkpoint; write its CSV / PR curve / panels. Returns the
    overall metrics (for the combined comparison)."""
    name = str(entry["name"])
    checkpoint = str(entry["checkpoint"])
    print(f"\n=== {name} ===\n  checkpoint: {checkpoint}")

    out_cfg = cfg.output
    metric_types = (
        list(out_cfg.metrics.types)
        if out_cfg.metrics.get("enabled", True)
        else []
    )
    do_pr = bool(out_cfg.pr_curves.get("enabled", True))
    do_roc = bool(out_cfg.get("roc_curves", {}).get("enabled", False))
    vis_cfg = out_cfg.visualizations
    n_vis = int(vis_cfg.get("n_samples", 0)) if vis_cfg.get("enabled", True) else 0

    # All outputs for this model go under a dedicated subdirectory.
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    model = load_model(checkpoint, device)
    subsets = list(cfg.data.val_subsets)

    # Per-subset + overall accumulation. Visualization panels are split evenly
    # across subsets so every subset contributes (roughly) the same number of
    # GT-vs-Pred panels instead of the first subset filling the whole budget.
    n_subsets = max(1, len(subsets))
    vis_quota = [
        n_vis // n_subsets + (1 if i < n_vis % n_subsets else 0)
        for i in range(n_subsets)
    ]
    rows: list[dict] = []
    pr_curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    roc_curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    overall_logits, overall_target, overall_mask = [], [], []
    overall_panels: list[tuple[np.ndarray, str, int]] = []

    for i, subset in enumerate(subsets):
        ds = build_val_dataset(cfg, [subset])
        if len(ds) == 0:
            print(f"  [skip] {subset}: 0 samples")
            continue
        loader = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 256)),
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=True,
        )
        need_panels = vis_quota[i]
        logits, target, mask, panels = run_inference(
            model, loader, device, need_panels
        )
        overall_logits.append(logits)
        overall_target.append(target)
        overall_mask.append(mask)
        overall_panels.extend((p, subset, img_id) for p, img_id in panels)

        if metric_types:
            m = compute_metrics(logits, target, mask, metric_types)
            row = {"subset": subset, "n_samples": len(ds), **m}
            rows.append(row)
            print(
                f"  {subset} (n={len(ds)}): "
                + "  ".join(f"{k}={m[k]:.4f}" for k in metric_types)
            )
        if do_pr:
            curve = compute_pr_curve(logits, target, mask)
            if curve is not None:
                pr_curves.append((subset, curve[0], curve[1]))
        if do_roc:
            rc = compute_roc_curve(logits, target, mask)
            if rc is not None:
                roc_curves.append((subset, rc[0], rc[1]))

    if not overall_logits:
        print(f"  [warn] no samples evaluated for {name}")
        return {"name": name, "n_samples": 0}

    logits_all = np.concatenate(overall_logits)
    target_all = np.concatenate(overall_target)
    mask_all = np.concatenate(overall_mask)
    n_all = logits_all.shape[0]

    overall_metrics: dict[str, float] = {}
    if metric_types:
        overall_metrics = compute_metrics(
            logits_all, target_all, mask_all, metric_types
        )
        rows.append({"subset": "overall", "n_samples": n_all, **overall_metrics})
        print(
            f"  overall (n={n_all}): "
            + "  ".join(f"{k}={overall_metrics[k]:.4f}" for k in metric_types)
        )
        if out_cfg.metrics.get("save", True):
            csv_path = os.path.join(model_dir, f"{name}.csv")
            save_metrics_csv(csv_path, metric_types, rows)
            print(f"  -> {csv_path}")

    overall_curve = None
    if do_pr:
        overall_curve = compute_pr_curve(logits_all, target_all, mask_all)
        if overall_curve is not None:
            pr_curves.append(("overall", overall_curve[0], overall_curve[1]))
        if out_cfg.pr_curves.get("save", True) and pr_curves:
            pr_path = os.path.join(model_dir, f"{name}_pr_curves.jpg")
            save_pr_curves(pr_path, pr_curves, f"{name} PR curve (micro-avg)")
            print(f"  -> {pr_path}")

    overall_roc = None
    if do_roc:
        overall_roc = compute_roc_curve(logits_all, target_all, mask_all)
        if overall_roc is not None:
            roc_curves.append(("overall", overall_roc[0], overall_roc[1]))
        if out_cfg.get("roc_curves", {}).get("save", True) and roc_curves:
            roc_path = os.path.join(model_dir, f"{name}_roc_curves.jpg")
            save_roc_curves(roc_path, roc_curves, f"{name} ROC curve (micro-avg)")
            print(f"  -> {roc_path}")

    if n_vis > 0 and vis_cfg.get("save", True) and overall_panels:
        save_vis_panels(model_dir, name, overall_panels)
        print(f"  -> {len(overall_panels)} visualization panels in {model_dir}")

    return {
        "name": name,
        "n_samples": n_all,
        **overall_metrics,
        "_pr_curve": overall_curve,
        "_roc_curve": overall_roc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = str(cfg.output.dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"device: {device}   output: {out_dir}")

    models = list(cfg.get("models", []))
    if not models:
        raise ValueError("config.models is empty")

    metric_types = (
        list(cfg.output.metrics.types)
        if cfg.output.metrics.get("enabled", True)
        else []
    )

    summaries = []
    for entry in models:
        summaries.append(evaluate_model(cfg, entry, device, out_dir))

    # Combined comparison across models.
    valid = [s for s in summaries if s.get("n_samples", 0) > 0]
    if len(valid) >= 1 and metric_types and cfg.output.metrics.get("save", True):
        cmp_path = os.path.join(out_dir, "comparison.csv")
        fieldnames = ["name", "n_samples"] + metric_types
        with open(cmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in valid:
                writer.writerow({k: s.get(k, "") for k in fieldnames})
        print(f"\ncomparison metrics -> {cmp_path}")

    if (
        len(valid) >= 2
        and cfg.output.pr_curves.get("enabled", True)
        and cfg.output.pr_curves.get("save", True)
    ):
        curves = [
            (s["name"], s["_pr_curve"][0], s["_pr_curve"][1])
            for s in valid
            if s.get("_pr_curve") is not None
        ]
        if curves:
            cmp_pr = os.path.join(out_dir, "comparison_pr_curves.jpg")
            save_pr_curves(cmp_pr, curves, "Model comparison PR curve (micro-avg)")
            print(f"comparison PR curves -> {cmp_pr}")

    if (
        len(valid) >= 2
        and cfg.output.get("roc_curves", {}).get("enabled", False)
        and cfg.output.get("roc_curves", {}).get("save", True)
    ):
        curves = [
            (s["name"], s["_roc_curve"][0], s["_roc_curve"][1])
            for s in valid
            if s.get("_roc_curve") is not None
        ]
        if curves:
            cmp_roc = os.path.join(out_dir, "comparison_roc_curves.jpg")
            save_roc_curves(
                cmp_roc, curves, "Model comparison ROC curve (micro-avg)"
            )
            print(f"comparison ROC curves -> {cmp_roc}")

    print("\nevaluation done.")


if __name__ == "__main__":
    main()
