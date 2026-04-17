"""Minimal demo: load image -> detect hands + visibility -> save annotated result."""

from __future__ import annotations

import argparse
import sys

import cv2
import torch

from hand_visibility_detector import HandVisibilityPipeline, draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand visibility detection demo")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output", default="output.jpg", help="Output path")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--hand-conf", type=float, default=0.5, help="Hand detection confidence threshold")
    parser.add_argument("--checkpoint", default=None, help="Path to visibility checkpoint (auto-downloads if omitted)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    dtype = torch.float32

    print("Loading pipeline...")
    pipe = HandVisibilityPipeline(
        device=device,
        dtype=dtype,
        vis_checkpoint=args.checkpoint,
        hand_conf=args.hand_conf,
    )

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image '{args.image}'", file=sys.stderr)
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Running inference...")
    results = pipe.predict(image_rgb)

    annotated = draw_detections(image_rgb, results)
    cv2.imwrite(args.output, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"Detected {len(results)} hand(s)")
    for i, r in enumerate(results):
        side = "Right" if r.is_right else "Left"
        vis_list = [round(float(v), 2) for v in r.visibility]
        print(f"  [{i}] {side} hand  conf={r.bbox_conf:.2f}  visibility={vis_list}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
