"""Minimal demo: load image -> detect hands + visibility -> save annotated result."""

from __future__ import annotations

import argparse
import sys

import cv2
import torch

from hand_visibility_detector import (
    HandVisibilityPipeline,
    draw_detections,
    fingertip_rotations,
    matrix_to_euler,
)
from hand_visibility_detector.rotations import FINGER_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand visibility detection demo")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("-o", "--output", default="output.jpg", help="Output path")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--hand-conf", type=float, default=0.5, help="Hand detection confidence threshold")
    parser.add_argument("--checkpoint", default=None, help="Path to visibility checkpoint (auto-downloads if omitted)")
    parser.add_argument("--backbone", default=None, help="Published checkpoint to auto-download when --checkpoint is omitted (wilor or hamer)")
    parser.add_argument("--show-global-orient", action="store_true", help="Visualize global_orient (wrist axes + roll/pitch/yaw text)")
    parser.add_argument("--show-hand-pose", action="store_true", help="Visualize hand_pose (per-joint rotation axes)")
    args = parser.parse_args()

    show_rotations = args.show_global_orient or args.show_hand_pose

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
        backbone=args.backbone,
        hand_conf=args.hand_conf,
    )

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image '{args.image}'", file=sys.stderr)
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Running inference...")
    results = pipe.predict(image_rgb, return_rotations=show_rotations)

    annotated = draw_detections(
        image_rgb,
        results,
        show_global_orient=args.show_global_orient,
        show_hand_pose=args.show_hand_pose,
    )
    cv2.imwrite(args.output, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"Detected {len(results)} hand(s)")
    for i, r in enumerate(results):
        side = "Right" if r.is_right else "Left"
        vis_list = [round(float(v), 2) for v in r.visibility]
        print(f"  [{i}] {side} hand  conf={r.bbox_conf:.2f}  visibility={vis_list}")
        if r.global_orient_euler is not None:
            roll, pitch, yaw = r.global_orient_euler
            print(f"      global_orient (deg): roll={roll:+.1f} pitch={pitch:+.1f} yaw={yaw:+.1f}")
        if args.show_hand_pose and r.hand_pose_euler is not None:
            for j, (jr, jp, jy) in enumerate(r.hand_pose_euler):
                print(f"      hand_pose[{j:2d}] (deg): roll={jr:+.1f} pitch={jp:+.1f} yaw={jy:+.1f}")
            tip_euler = matrix_to_euler(
                fingertip_rotations(r.global_orient, r.hand_pose)
            )
            for name, (tr, tp, ty) in zip(FINGER_NAMES, tip_euler):
                print(f"      fingertip {name:6s} (deg): roll={tr:+.1f} pitch={tp:+.1f} yaw={ty:+.1f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
