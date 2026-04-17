"""hand-visibility-detector: Per-keypoint hand visibility detection using WiLoR-mini."""

import warnings as _warnings

# Silence known upstream deprecation noise (timm, smplx/numpy, torch.load weights_only)
_warnings.filterwarnings("ignore", message=r".*weights_only=False.*", category=FutureWarning)
_warnings.filterwarnings("ignore", message=r".*timm\.models\.layers.*", category=FutureWarning)
_warnings.filterwarnings("ignore", message=r".*align=0.*")

from .pipeline import HandResult, HandVisibilityPipeline
from .visualization import draw_detections, vis_color

__version__ = "0.1.0"

__all__ = [
    "HandVisibilityPipeline",
    "HandResult",
    "draw_detections",
    "vis_color",
]
