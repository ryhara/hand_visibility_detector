# Hand Visibility Detector

Per-keypoint hand visibility detection built on top of [WiLoR-mini](https://github.com/warmshao/WiLoR-mini).
Given an RGB image, the pipeline detects both hands and predicts a visibility score for each of the 21 MANO keypoints.

Pretrained weights are hosted on HuggingFace Hub: **[ryhara/hand-visibility-detector](https://huggingface.co/ryhara/hand-visibility-detector)**

## Update
- [ ] add training code
- [x] 2026/04/17 publish to github

## Installation

### As a dependency (import from your project)

```bash
uv add git+https://github.com/ryhara/hand_visibility_detector.git
# with the Gradio demo extras
uv add "hand-visibility-detector[demo] @ git+https://github.com/ryhara/hand_visibility_detector.git"
```

Or with `pip`:

```bash
pip install git+https://github.com/ryhara/hand_visibility_detector.git
pip install "hand-visibility-detector[demo] @ git+https://github.com/ryhara/hand_visibility_detector.git"
```

### Running this repository locally (clone & run demo)

```bash
git clone https://github.com/ryhara/hand_visibility_detector.git
cd hand_visibility_detector
uv sync                  # base deps
uv sync --extra demo     # + Gradio demo deps
```

## Demo

CLI:

```bash
python demo.py path/to/image.jpg -o output.jpg
```

Gradio UI:

```bash
python demo_gradio.py
```

## Dataset

- [COCO-WholeBody Dataset](https://github.com/jin-s13/COCO-WholeBody)
    ```bash
    curl -LO http://images.cocodataset.org/zips/train2017.zip
    curl -LO http://images.cocodataset.org/zips/val2017.zip
    curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    ```

    ```bash
    COCO-WholeBody/
    ├── annotations
    │   ├── coco_wholebody_train_v1.0.json
    │   └── coco_wholebody_val_v1.0.json
    ├── train2017
    │   ├── XXXXXXXXXXXX.jpg
    │   └── ...
    └── val2017
        ├── XXXXXXXXXXXX.jpg
        └── ...
    ```

- [HInt Dataset](https://github.com/ddshan/hint)
    ```bash
    curl -LO https://fouheylab.eecs.umich.edu/~dandans/projects/hammer/HInt_annotation_partial.zip
    ```

    *: I don't use the ego4d dataset. Because we need to download the ego4d dataset from the official website. Check the [HInt Dataset](https://github.com/ddshan/hint) for more details.

    ```bash
    HInt_annotation_partial/
    ├── TEST_ego4d_img*
    ├── TEST_ego4d_seq*
    ├── TEST_epick_img
    ├── TEST_newdays_img
    ├── TRAIN_ego4d_img*
    ├── TRAIN_epick_img
    ├── TRAIN_newdays_img
    ├── VAL_ego4d_img*
    ├── VAL_ego4d_seq*
    ├── VAL_epick_img
    └── VAL_newdays_img
    ```

## License

This project is released for **research and non-commercial use only**, inheriting the most restrictive terms of its upstream dependencies. Any use of this code, weights, or derivatives must comply with **all** of the following:

- **[COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody)**
- **[HInt](https://github.com/ddshan/hint)**
- **[WiLoR](https://github.com/rolpotamias/WiLoR)** (and [WiLoR-mini](https://github.com/warmshao/WiLoR-mini))
- **[MANO](https://mano.is.tue.mpg.de/)**
- **[Ultralytics](https://ultralytics.com/)**

## Citation

If you use this software, please cite it as:

```bibtex
@software{hara2026handvisibility,
  author  = {Hara, Ryosei},
  title   = {Hand Visibility Detector},
  year    = {2026},
  url     = {https://github.com/ryhara/hand_visibility_detector},
  version = {0.1.0}
}
```