"""Gradio demo: interactive hand visibility detection on images and videos."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio hand visibility demo")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--checkpoint", default=None, help="Visibility checkpoint path")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    import gradio as gr

    from hand_visibility_detector import HandVisibilityPipeline, draw_detections

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float32

    pipe = HandVisibilityPipeline(
        device=device,
        dtype=dtype,
        vis_checkpoint=args.checkpoint,
    )

    def process_image(
        image_rgb: np.ndarray,
        hand_conf: float,
        show_bones: bool,
    ) -> tuple[np.ndarray, str]:
        if image_rgb is None:
            return np.zeros((256, 256, 3), dtype=np.uint8), "No image provided"
        pipe.hand_conf = hand_conf
        results = pipe.predict(image_rgb)
        annotated = draw_detections(image_rgb, results, show_bones=show_bones)
        info_lines = [f"Detected {len(results)} hand(s)"]
        for i, r in enumerate(results):
            side = "R" if r.is_right else "L"
            avg_vis = r.visibility.mean()
            info_lines.append(
                f"  [{i}] {side}  conf={r.bbox_conf:.2f}  avg_vis={avg_vis:.2f}"
            )
        return annotated, "\n".join(info_lines)

    def process_video(
        video_path: str,
        hand_conf: float,
        show_bones: bool,
        progress: gr.Progress = gr.Progress(),
    ) -> str | None:
        if video_path is None:
            return None
        import imageio

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        out_path = tempfile.mktemp(suffix=".mp4")
        pipe.hand_conf = hand_conf

        progress(0.0, desc="Starting...")
        with imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        ) as writer:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pipe.predict(frame_rgb)
                annotated = draw_detections(frame_rgb, results, show_bones=show_bones)
                writer.append_data(annotated)
                frame_idx += 1
                if total > 0:
                    progress(frame_idx / total, desc=f"Frame {frame_idx}/{total}")
                else:
                    progress(0.5, desc=f"Frame {frame_idx}")

        cap.release()
        progress(1.0, desc="Done")
        return out_path

    with gr.Blocks(title="Hand Visibility Detector") as demo:
        gr.Markdown("## Hand Visibility Detector")
        gr.Markdown(
            "Detect hands, estimate 3D pose (WiLoR-mini), and predict "
            "per-keypoint visibility. Green = visible, Red = occluded."
        )

        with gr.Row():
            hand_conf_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Hand detection confidence",
            )
            show_bones_cb = gr.Checkbox(value=True, label="Show bones")

        with gr.Tabs():
            # -- Single image tab --
            with gr.Tab("Single Image"):
                with gr.Row():
                    img_input = gr.Image(label="Input", type="numpy")
                    img_output = gr.Image(label="Result", type="numpy")
                img_info = gr.Textbox(label="Info", interactive=False)
                img_btn = gr.Button("Detect")
                img_btn.click(
                    fn=process_image,
                    inputs=[img_input, hand_conf_slider, show_bones_cb],
                    outputs=[img_output, img_info],
                )

            # -- Video tab --
            with gr.Tab("Video"):
                vid_input = gr.Video(label="Input video")
                vid_output = gr.Video(label="Result video")
                vid_btn = gr.Button("Process Video")
                vid_btn.click(
                    fn=process_video,
                    inputs=[vid_input, hand_conf_slider, show_bones_cb],
                    outputs=[vid_output],
                )

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
