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
