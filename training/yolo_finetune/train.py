"""Fine-tune the WiLoR hand detector on HInt.

Loads the pretrained `detector.pt` from the `warmshao/WiLoR-mini` HuggingFace
repo (the same checkpoint that
`wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline.init_models` uses)
and continues YOLO training on the HInt dataset prepared by
`prepare_dataset.py`.

Class indices match WiLoR's pipeline:
    0 = left hand, 1 = right hand

After training finishes, the script evaluates on the `test` split
(epick + newdays; ego4d is excluded from the prepared dataset).


Settings rationale (vs. the original WiLoR paper, §5.1 of arxiv:2409.12259)
==========================================================================
The original detector was trained from scratch on WHIM (~2M images) for 200
epochs with batch=256 on 2× RTX 4090 over ~3 weeks.  Here we *fine-tune* a
small HInt subset (~12k train images, epick+newdays), so some hyperparameters
are intentionally relaxed.  Below: "matched" = same as the paper, "relaxed" =
softened for fine-tuning.

  optimizer       Adam                                   [matched]
  loss box=15, cls=0.5, dfl=1.5                          [matched]
  rotation ±60°  (`degrees=60`)                          [matched]
  scale jitter ±0.5 (`scale=0.5`, range [0.5, 1.5])      [matched]
  mosaic prob 0.7                                        [matched]
  mixup enabled (low prob; paper just says "follows")    [matched, low value]
  fliplr = 0                                             [matched: left/right
                                                          classes; horizontal
                                                          flip would swap them
                                                          without label swap]
  flipud = 0                                             [matched]
  lr0             1e-3 (paper: 0.01 scratch)             [relaxed for FT]
  lrf             0.01 (final lr = lr0*lrf ≈ 1e-5)       [relaxed]
  epochs          50 (paper: 200)                        [relaxed]
  patience        15 (paper: 30)                         [relaxed]
  batch           16  (paper: 256 on 2×4090)             [hardware dependent]
  imgsz           640 (paper does not specify)           [Ultralytics default]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics import settings as ultra_settings


WILOR_MINI_REPO_ID = "warmshao/WiLoR-mini"


def resolve_pretrained(weights: str | None, cache_dir: Path) -> str:
    """Return a path to detector.pt, downloading from HF if not provided."""
    if weights:
        p = Path(weights)
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)

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
    return str(local)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "hint_yolo" / "data.yaml",
        help="Path to the data.yaml produced by prepare_dataset.py",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to detector.pt; if omitted, downloads from HuggingFace",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "wilor_cache",
        help="Where to cache the downloaded detector.pt",
    )
    ap.add_argument("--project", type=str, default="runs/hint_yolo")
    ap.add_argument("--name", type=str, default="finetune")

    # --- schedule (relaxed for fine-tuning) ---
    ap.add_argument("--epochs", type=int, default=50,
                    help="paper used 200 (scratch); 50 is plenty for FT")
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stopping patience; paper used 30 (scratch)")
    ap.add_argument("--lr0", type=float, default=1e-3,
                    help="paper used 0.01 (scratch); lowered for FT to avoid forgetting")
    ap.add_argument("--lrf", type=float, default=0.01,
                    help="final LR factor: final_lr = lr0 * lrf  (paper: linear decay to 1e-6)")
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--cos-lr", action="store_true", default=False,
                    help="cosine schedule; default is linear (matches paper)")

    # --- hardware ---
    ap.add_argument("--batch", type=int, default=16,
                    help="paper used 256 on 2x RTX 4090; pick what fits your GPU")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="paper does not specify; 640 is the Ultralytics default")
    ap.add_argument("--device", type=str, default="0",
                    help="cuda device id, 'cpu', or comma-separated for multi-GPU")
    ap.add_argument("--workers", type=int, default=8)

    # --- augmentation (matched to paper) ---
    ap.add_argument("--mosaic", type=float, default=0.7, help="paper §5.1")
    ap.add_argument("--mixup",  type=float, default=0.1,
                    help="paper says mixup is used but does not give a value")
    ap.add_argument("--degrees", type=float, default=60.0, help="paper §5.1: ±60°")
    ap.add_argument("--scale",   type=float, default=0.5,
                    help="paper §5.1: random scaling in [0.5, 1] (Ultralytics interprets as ±0.5)")
    ap.add_argument("--fliplr",  type=float, default=0.0,
                    help="MUST be 0: classes are left vs right; flipping without swap breaks labels")
    ap.add_argument("--flipud",  type=float, default=0.0)
    ap.add_argument("--erasing", type=float, default=0.4,
                    help="paper mentions random masking; Ultralytics' `erasing` covers it")
    ap.add_argument("--close-mosaic", type=int, default=10,
                    help="disable mosaic for the last N epochs (Ultralytics convention)")

    # --- loss weights (matched to paper §5.1) ---
    ap.add_argument("--box", type=float, default=15.0, help="paper §5.1 λ₂=15")
    ap.add_argument("--cls", type=float, default=0.5,  help="paper §5.1 λ₀=0.5")
    ap.add_argument("--dfl", type=float, default=1.5,  help="paper §5.1 λ₁=1.5")

    # --- misc ---
    ap.add_argument("--optimizer", type=str, default="Adam",
                    help="paper §5.1: Adam (Ultralytics default is auto/SGD)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-test", action="store_true")

    # --- W&B ---
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                    help="enable Weights & Biases logging (default: on)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false",
                    help="disable W&B logging")
    ap.add_argument("--wandb-project", type=str, default="hand-visibility",
                    help="W&B project name")
    ap.add_argument("--wandb-name", type=str, default=None,
                    help="W&B run name (defaults to --name)")
    ap.add_argument("--wandb-entity", type=str, default=None,
                    help="W&B entity (team/user); defaults to wandb default")
    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} not found. Run prepare_dataset.py first."
        )
    if args.fliplr != 0.0:
        # Loud warning rather than silent corruption.
        print(
            f"[WARN] fliplr={args.fliplr} != 0. Left/right hand classes will be "
            "mislabeled on flipped samples (Ultralytics does not swap class ids "
            "on horizontal flip). You almost certainly want fliplr=0.0."
        )

    weights_path = resolve_pretrained(args.weights, args.cache_dir)
    print(f"initial weights: {weights_path}")

    wandb_run_id: str | None = None
    if args.wandb:
        ultra_settings.update({"wandb": True})
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or args.name,
            entity=args.wandb_entity,
            config=vars(args),
            resume="allow" if args.resume else None,
        )
        wandb_run_id = run.id if run is not None else None

    model = YOLO(weights_path)

    train_results = model.train(
        data=str(args.data),
        # schedule
        epochs=args.epochs,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        seed=args.seed,
        # hardware
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        # augmentation
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        erasing=args.erasing,
        close_mosaic=args.close_mosaic,
        # loss weights (paper §5.1)
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        # bookkeeping
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
    )
    print("training done.")
    try:
        save_dir = Path(train_results.save_dir)
    except AttributeError:
        save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"best weights: {best}")

    if args.skip_test:
        if args.wandb and wandb.run is not None:
            wandb.finish()
        return
    if not best.exists():
        print(f"[warn] best.pt not found at {best}; skipping test eval.")
        if args.wandb and wandb.run is not None:
            wandb.finish()
        return

    # Evaluate the baseline (WiLoR detector.pt) AND the fine-tuned best.pt on
    # the test split, then print a side-by-side comparison with Δ.
    from evaluate import evaluate_one, print_comparison

    results: dict[str, dict[str, float]] = {}
    results["baseline"] = evaluate_one(
        label="baseline",
        weights=Path(weights_path),
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name_suffix=f"{args.name}_test",
    )
    results["finetuned"] = evaluate_one(
        label="finetuned",
        weights=best,
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name_suffix=f"{args.name}_test",
    )
    print_comparison(results)

    if args.wandb:
        flat = {f"test/{label}/{k}": v
                for label, metrics in results.items()
                for k, v in metrics.items()}
        if "baseline" in results and "finetuned" in results:
            for k in results["finetuned"]:
                if k in results["baseline"]:
                    flat[f"test/delta/{k}"] = results["finetuned"][k] - results["baseline"][k]
        # Ultralytics' built-in W&B callback calls wandb.finish() at the end of
        # training, so wandb.run is None here. Resume the same run to attach the
        # test metrics to it.
        if wandb.run is None and wandb_run_id is not None:
            # resume="must": 既存runに追記。同じrun_idなのでサーバ上の
            # 学習中metrics/config/summaryはそのまま保持される(初期化されない)。
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=wandb_run_id,
                resume="must",
            )
        wandb.log(flat)
        wandb.finish()


if __name__ == "__main__":
    main()
