"""Train HandVisibilityNet on COCO-WholeBody or HInt hand annotations.

Setup:
    * WiLoR ViT backbone (optionally frozen) + RTMPose-style visibility head
      (conv-FC-GAU-conv), matching the inference package.
    * AdamW, LinearLR warmup + CosineAnnealingLR
    * Effective batch size via gradient accumulation
    * Gradient clipping, AMP, optional masked-BCE w/ pos_weight

Usage:
    python -m training.train \
        --config training/configs/hint.yaml \
        data.hint_root=/path/to/HInt_annotation_partial

Override any config field via dotted CLI args (OmegaConf style).
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import COCOWholeBodyHandDataset, HIntHandDataset
from .model import build_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    logits: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """``logits`` / ``target`` / ``mask``: ``(N, 21)`` numpy arrays."""
    from sklearn.metrics import average_precision_score

    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(np.float32)

    valid = mask > 0.5
    correct = ((pred == target) & valid).sum()
    total = valid.sum()
    accuracy = float(correct / max(total, 1))

    aps = []
    for k in range(target.shape[1]):
        v = mask[:, k] > 0.5
        if v.sum() < 2:
            continue
        t_k = target[v, k]
        if t_k.sum() == 0 or t_k.sum() == t_k.shape[0]:
            continue
        aps.append(average_precision_score(t_k, probs[v, k]))
    mAP = float(np.mean(aps)) if aps else 0.0

    tp = ((pred == 1) & (target == 1) & valid).sum()
    fp = ((pred == 1) & (target == 0) & valid).sum()
    fn = ((pred == 0) & (target == 1) & valid).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"acc": accuracy, "mAP": float(mAP), "f1": float(f1)}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, target)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    all_logits, all_target, all_mask = [], [], []
    losses = []
    for batch in tqdm(loader, desc="val", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(img)
        loss = masked_bce(logits, target, mask, pos_weight)
        losses.append(loss.item())
        all_logits.append(logits.detach().float().cpu().numpy())
        all_target.append(target.cpu().numpy())
        all_mask.append(mask.cpu().numpy())
    metrics = compute_metrics(
        np.concatenate(all_logits),
        np.concatenate(all_target),
        np.concatenate(all_mask),
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def estimate_pos_weight(
    dataset: COCOWholeBodyHandDataset | HIntHandDataset,
) -> torch.Tensor:
    pos, tot = dataset.positive_counts()
    neg = tot - pos
    weight = np.where(pos > 0, neg / np.clip(pos, 1, None), 1.0)
    weight = np.clip(weight, 0.25, 10.0)
    return torch.tensor(weight, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    warmup_start_factor: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


_USE_WANDB = False


def _log(data: dict, step: int) -> None:
    if _USE_WANDB:
        import wandb

        wandb.log(data, step=step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataset(cfg, train: bool):
    augment_cfg_node = cfg.data.get("augment", None)
    augment_cfg = (
        OmegaConf.to_container(augment_cfg_node, resolve=True)
        if augment_cfg_node is not None
        else {}
    )
    dataset_type = str(cfg.data.get("type", "cocowholebody")).lower()
    if dataset_type == "cocowholebody":
        return COCOWholeBodyHandDataset(
            ann_path=cfg.data.train_ann if train else cfg.data.val_ann,
            img_dir=cfg.data.train_img_dir if train else cfg.data.val_img_dir,
            crop_size=cfg.data.crop_size,
            bbox_expand=cfg.data.bbox_expand,
            min_bbox_side=cfg.data.min_bbox_side,
            train=train,
            augment_cfg=augment_cfg if train else None,
        )
    if dataset_type == "hint":
        subsets = cfg.data.train_subsets if train else cfg.data.val_subsets
        return HIntHandDataset(
            data_root=cfg.data.hint_root,
            subsets=list(subsets),
            crop_size=cfg.data.crop_size,
            bbox_expand=cfg.data.bbox_expand,
            min_bbox_side=cfg.data.min_bbox_side,
            train=train,
            augment_cfg=augment_cfg if train else None,
        )
    raise ValueError(f"unknown data.type: {dataset_type}")


def main() -> None:
    global _USE_WANDB

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)

    seed = int(cfg.train.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.train.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))

    train_ds = build_dataset(cfg, train=True)
    val_ds = build_dataset(cfg, train=False)
    dataset_type = str(cfg.data.get("type", "cocowholebody")).lower()
    print(f"[{dataset_type}] train hands: {len(train_ds)}   val hands: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(cfg.train.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.train.num_workers) > 0,
    )

    # ── pos_weight ────────────────────────────────────────────────────────
    pos_weight: torch.Tensor | None
    pw_cfg = cfg.train.pos_weight
    if pw_cfg is None:
        pos_weight = None
    elif pw_cfg == "auto":
        pos_weight = estimate_pos_weight(train_ds)
        print(f"estimated pos_weight: {pos_weight.tolist()}")
    else:
        pos_weight = torch.tensor(list(pw_cfg), dtype=torch.float32)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    # ── Model / optim ─────────────────────────────────────────────────────
    freeze_backbone = bool(cfg.model.get("freeze_backbone", False))
    model = build_model(
        dropout=float(cfg.model.dropout),
        hidden_dim=int(cfg.model.get("hidden_dim", 256)),
        freeze_backbone=freeze_backbone,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if freeze_backbone:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"freeze_backbone=True  trainable params: "
            f"{n_trainable / 1e6:.2f}M / total: {n_total / 1e6:.2f}M"
        )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=int(cfg.train.epochs),
        warmup_epochs=int(cfg.train.warmup_epochs),
        warmup_start_factor=float(cfg.train.warmup_start_factor),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))

    grad_accum = max(1, int(cfg.train.grad_accum_steps))
    grad_clip = float(cfg.train.grad_clip) if cfg.train.get("grad_clip") else 0.0

    start_epoch = 0
    best_map = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if ckpt.get("head_only", False):
            model.load_head_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        print(f"resumed from {args.resume} (epoch {start_epoch})")

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    _USE_WANDB = bool(wandb_cfg.get("enabled", True))
    if _USE_WANDB:
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "hand-visibility"),
            name=wandb_cfg.get("name") or os.path.basename(out_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=out_dir,
            resume="allow" if args.resume else None,
        )

    eff_batch = int(cfg.train.batch_size) * grad_accum
    print(
        f"effective batch = {int(cfg.train.batch_size)} x {grad_accum} = {eff_batch}"
    )

    # ── Training loop ─────────────────────────────────────────────────────
    total_epochs = int(cfg.train.epochs)
    global_step = 0

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{total_epochs}",
            leave=False,
            dynamic_ncols=True,
        )
        for it, batch in enumerate(pbar):
            img = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp)):
                logits = model(img)
                loss = masked_bce(logits, target, mask, pos_weight)
                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()

            is_step = ((it + 1) % grad_accum == 0) or (it + 1 == len(train_loader))
            if is_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss_sum += loss.item() * img.shape[0]
            epoch_n += img.shape[0]
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if (it + 1) % int(cfg.train.log_every) == 0:
                lr = optimizer.param_groups[0]["lr"]
                _log(
                    {"train/loss_iter": loss.item(), "train/lr": lr},
                    step=global_step,
                )

        scheduler.step()
        train_loss = epoch_loss_sum / max(epoch_n, 1)
        _log({"train/loss_epoch": train_loss, "epoch": epoch}, step=global_step)
        tqdm.write(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"time={time.time() - t0:.1f}s"
        )

        if (epoch + 1) % int(cfg.train.val_every_epoch) == 0 or (
            epoch + 1 == total_epochs
        ):
            metrics = evaluate(model, val_loader, device, pos_weight)
            tqdm.write(
                f"[epoch {epoch}] val_loss={metrics['loss']:.4f} "
                f"acc={metrics['acc']:.4f} mAP={metrics['mAP']:.4f} "
                f"f1={metrics['f1']:.4f}"
            )
            _log(
                {f"val/{k}": v for k, v in metrics.items()} | {"epoch": epoch},
                step=global_step,
            )

            model_sd = (
                model.head_state_dict() if freeze_backbone else model.state_dict()
            )
            ckpt = {
                "model": model_sd,
                "head_only": freeze_backbone,
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(ckpt, os.path.join(out_dir, "ckpt", "last.pt"))
            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                ckpt["best_map"] = best_map
                torch.save(ckpt, os.path.join(out_dir, "ckpt", "best.pt"))
                tqdm.write(f"  -> new best mAP {best_map:.4f} saved")

    if _USE_WANDB:
        import wandb

        wandb.finish()
    print("training done.")


if __name__ == "__main__":
    main()
