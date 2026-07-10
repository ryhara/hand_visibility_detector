"""Video demo: read video -> detect hands + visibility per frame -> save annotated video."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import torch

from hand_visibility_detector import HandVisibilityPipeline, draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand visibility detection video demo")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video path")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--hand-conf", type=float, default=0.5, help="Hand detection confidence threshold")
    parser.add_argument("--checkpoint", default=None, help="Path to visibility checkpoint (auto-downloads if omitted)")
    parser.add_argument("--backbone", default=None, help="Published checkpoint to auto-download when --checkpoint is omitted (wilor or hamer)")
    parser.add_argument("--show-global-orient", action="store_true", help="Visualize global_orient (wrist axes + roll/pitch/yaw text)")
    parser.add_argument("--show-hand-pose", action="store_true", help="Visualize hand_pose (per-joint rotation axes)")
    parser.add_argument("--frame-skip", type=int, default=0, help="Process every (N+1)-th frame; skipped frames reuse the last result")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing this many frames")
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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.video}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: cannot open video writer for '{args.output}'", file=sys.stderr)
        cap.release()
        sys.exit(1)

    print(f"Input: {width}x{height} @ {fps:.1f} fps, {total} frames")
    print("Running inference...")

    frame_idx = 0
    processed = 0
    results = []
    start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if args.frame_skip <= 0 or frame_idx % (args.frame_skip + 1) == 0:
                results = pipe.predict(frame_rgb, return_rotations=show_rotations)
                processed += 1

            annotated = draw_detections(
                frame_rgb,
                results,
                show_global_orient=args.show_global_orient,
                show_hand_pose=args.show_hand_pose,
            )
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                progress = f"{frame_idx}/{total}" if total > 0 else str(frame_idx)
                print(f"  frame {progress}  ({proc_fps:.1f} fps)")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start
    print(f"Done: {frame_idx} frames written ({processed} inferred) in {elapsed:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
