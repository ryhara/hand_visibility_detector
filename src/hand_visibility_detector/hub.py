"""HuggingFace Hub helpers for downloading visibility head checkpoints."""

from __future__ import annotations

HF_REPO_ID = "ryhara/hand-visibility-detector"
DEFAULT_CHECKPOINT = "best.pt"


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
