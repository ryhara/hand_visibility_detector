"""HuggingFace Hub helpers for downloading visibility head checkpoints."""

from __future__ import annotations

HF_REPO_ID = "ryhara/hand-visibility-detector"
DEFAULT_CHECKPOINT = "best.pt"

# Visibility checkpoints published on the Hub, keyed by backbone name. Other
# backbones (resnet / vit / cspnext / dinov2 / dinov3) have no published
# checkpoint, so a local ``vis_checkpoint`` path must be passed explicitly.
BACKBONE_CHECKPOINTS: dict[str, str] = {
    "wilor": "best.pt",
    "hamer": "best_hamer.pt",
}

# Official HaMeR demo Space; hosts the (otherwise gated) ``hamer.ckpt`` whose
# ``backbone.``-prefixed weights the HaMeR backbone loads.
HAMER_SPACE_ID = "geopavlakos/HaMeR"
HAMER_CKPT_FILE = "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"

# WiLoR-mini assets (YOLO hand detector, MANO, WiLoR weights). Cached inside the
# installed ``wilor_mini`` package, exactly where ``wilor_mini`` itself puts them.
WILOR_MINI_REPO_ID = "warmshao/WiLoR-mini"

# AnyHand (Si et al. 2026): WiLoR / HaMeR fine-tuned with the AnyHand synthetic
# dataset. Drop-in replacements for the original checkpoints (CC-BY-NC-ND-4.0).
ANYHAND_REPO_ID = "chen-si-02/AnyHand-Models"
ANYHAND_CHECKPOINTS: dict[str, str] = {
    "anyhand_wilor": "anyhand_wilor.ckpt",
    "anyhand_hamer": "anyhand_hamer.ckpt",
}


def download_checkpoint(
    filename: str = DEFAULT_CHECKPOINT,
    repo_id: str = HF_REPO_ID,
    cache_dir: str | None = None,
) -> str:
    """Download a visibility head checkpoint from HuggingFace Hub.

    Returns the local file path to the downloaded checkpoint.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )


def default_checkpoint_for_backbone(backbone: str) -> str:
    """Download the published visibility checkpoint for ``backbone``.

    Only backbones listed in :data:`BACKBONE_CHECKPOINTS` (``wilor`` /
    ``hamer``) have published checkpoints; anything else raises ``ValueError``
    asking for an explicit ``vis_checkpoint`` path.
    """
    key = backbone.lower()
    if key not in BACKBONE_CHECKPOINTS:
        raise ValueError(
            f"no published visibility checkpoint for backbone {backbone!r} "
            f"(published: {', '.join(sorted(BACKBONE_CHECKPOINTS))}). "
            "Pass vis_checkpoint=<path to your trained checkpoint> instead."
        )
    return download_checkpoint(BACKBONE_CHECKPOINTS[key])


def download_hamer_backbone(cache_dir: str | None = None) -> str:
    """Download HaMeR's ``hamer.ckpt`` from the official HaMeR demo Space.

    Returns the local file path to the cached checkpoint (~2.5 GB; the HaMeR
    backbone reads only the ``backbone.``-prefixed weights from it).
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=HAMER_SPACE_ID,
        repo_type="space",
        filename=HAMER_CKPT_FILE,
        cache_dir=cache_dir,
    )


def wilor_mini_asset(filename: str) -> str:
    """Return the local path of a WiLoR-mini asset (``detector.pt``,
    ``MANO_RIGHT.pkl``, ``mano_mean_params.npz``, ``wilor_final.ckpt``),
    downloading it into ``wilor_mini/pretrained_models/`` if missing."""
    import os

    import wilor_mini

    package_dir = os.path.dirname(wilor_mini.__file__)
    path = os.path.join(package_dir, "pretrained_models", filename)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=WILOR_MINI_REPO_ID,
            subfolder="pretrained_models",
            filename=filename,
            local_dir=package_dir,
        )
    return path


def download_anyhand_checkpoint(
    pose_model: str,
    cache_dir: str | None = None,
) -> str:
    """Download an AnyHand checkpoint (``anyhand_wilor`` / ``anyhand_hamer``,
    ~2.5 GB each) from HuggingFace Hub and return its local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=ANYHAND_REPO_ID,
        filename=ANYHAND_CHECKPOINTS[pose_model],
        cache_dir=cache_dir,
    )
