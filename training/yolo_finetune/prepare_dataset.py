"""HInt -> YOLO dataset converter.

Builds an Ultralytics YOLO-format dataset under <out_dir>:

    <out_dir>/
        images/{train,val,test}/*.jpg     (symlinks to HInt jpgs)
        labels/{train,val,test}/*.txt     (YOLO bbox labels)
        data.yaml

HInt source layout:
    /path/to/HInt_annotation_partial/{TRAIN,VAL,TEST}_{epick,newdays,ego4d}_img/<stem>_{r,l}.{jpg,json}

The ego4d splits are intentionally excluded (frames are not provided in the
partial release).  Each HInt json describes exactly one hand:
    [{"bbox": [[x1, y1, x2, y2]], "keypoints": [[u,v], ...], ...}]
Handedness is encoded in the file-name suffix (_r / _l), matching WiLoR's
detector convention: cls 0 = left hand, cls 1 = right hand.

Note: HInt only labels a single hand per frame; the other hand (if visible) is
unlabeled and will be treated as background by YOLO.  This is a dataset
limitation we accept.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# WiLoR-mini detector class indices (see wilor_hand_pose3d_estimation_pipeline.py:
# `is_rights.append(det.boxes.cls...)` is consumed as `right = is_rights[i]`,
# `flip = right == 0`).  So cls 0 == left, cls 1 == right.
CLASS_NAMES = ["left", "right"]

# WiLoR's detector.pt is a YOLO *pose* model with 21 hand keypoints (MANO order),
# so labels must include keypoints to match the model head.
N_KEYPOINTS = 21

DEFAULT_HINT_ROOT = Path("/path/to/HInt_annotation_partial")

# split_name -> [HInt subdir names]; ego4d intentionally excluded
SPLITS: dict[str, list[str]] = {
    "train": ["TRAIN_epick_img", "TRAIN_newdays_img"],
    "val":   ["VAL_epick_img",   "VAL_newdays_img"],
    "test":  ["TEST_epick_img",  "TEST_newdays_img"],
}


@dataclass
class Stats:
    n_images: int = 0
    n_labels: int = 0
    n_skipped_no_json: int = 0
    n_skipped_bad_bbox: int = 0


def parse_handedness(stem: str) -> int:
    """Return class index from filename stem ending in _r or _l."""
    suf = stem[-2:]
    if suf == "_r":
        return 1  # right
    if suf == "_l":
        return 0  # left
    raise ValueError(f"unexpected stem suffix: {stem!r}")


def hint_bbox_to_yolo(bbox_xyxy: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Convert absolute xyxy -> normalized xywh (YOLO).  Returns None if degenerate."""
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0.0, min(float(x1), img_w - 1))
    y1 = max(0.0, min(float(y1), img_h - 1))
    x2 = max(0.0, min(float(x2), img_w - 1))
    y2 = max(0.0, min(float(y2), img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def hint_keypoints_to_yolo(
    keypoints: list[list[float]],
    existence: list[float] | None,
    occlusion: list[float] | None,
    img_w: int,
    img_h: int,
) -> list[tuple[float, float, int]]:
    """Convert HInt keypoints to YOLO pose triplets (x_norm, y_norm, v).

    YOLO visibility flag:
      0 = not labeled (skipped by loss),
      1 = labeled but not visible (occluded),
      2 = labeled and visible.

    HInt semantics:
      existence[i] == 1 -> the joint is annotated in the frame,
      occlusion[i] == 1 -> annotated joint is occluded.
    If the (x, y) falls outside the image we treat it as unlabeled so Ultralytics
    doesn't see invalid normalized coords.
    """
    triplets: list[tuple[float, float, int]] = []
    n = len(keypoints)
    for i in range(n):
        x, y = float(keypoints[i][0]), float(keypoints[i][1])
        exists = bool(existence[i]) if existence is not None and i < len(existence) else True
        occluded = bool(occlusion[i]) if occlusion is not None and i < len(occlusion) else False

        in_bounds = 0.0 <= x <= img_w - 1 and 0.0 <= y <= img_h - 1
        if not exists or not in_bounds:
            v = 0
            xn, yn = 0.0, 0.0
        else:
            v = 1 if occluded else 2
            xn = x / img_w
            yn = y / img_h
            # Clamp for the rare case where normalized falls slightly outside [0, 1].
            xn = min(max(xn, 0.0), 1.0)
            yn = min(max(yn, 0.0), 1.0)
        triplets.append((xn, yn, v))
    return triplets


def process_split(
    src_dirs: list[Path],
    out_images: Path,
    out_labels: Path,
    *,
    link_mode: str,
) -> Stats:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    stats = Stats()

    for src_dir in src_dirs:
        if not src_dir.exists():
            print(f"[warn] missing source dir: {src_dir}")
            continue
        jpgs = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".jpg")
        for jpg in tqdm(jpgs, desc=f"{src_dir.name}"):
            stem = jpg.stem
            json_path = src_dir / f"{stem}.json"
            if not json_path.exists():
                stats.n_skipped_no_json += 1
                continue
            try:
                anns = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                stats.n_skipped_no_json += 1
                continue
            if not isinstance(anns, list) or len(anns) == 0:
                stats.n_skipped_no_json += 1
                continue
            ann = anns[0]
            bbox_field = ann.get("bbox")
            if (
                not bbox_field
                or not isinstance(bbox_field, list)
                or len(bbox_field) == 0
                or len(bbox_field[0]) != 4
            ):
                stats.n_skipped_bad_bbox += 1
                continue

            kpts_field = ann.get("keypoints")
            if (
                not isinstance(kpts_field, list)
                or len(kpts_field) != N_KEYPOINTS
                or any(not isinstance(p, list) or len(p) < 2 for p in kpts_field)
            ):
                stats.n_skipped_bad_bbox += 1
                continue

            cls = parse_handedness(stem)
            with Image.open(jpg) as im:
                w, h = im.size
            yolo_box = hint_bbox_to_yolo(bbox_field[0], w, h)
            if yolo_box is None:
                stats.n_skipped_bad_bbox += 1
                continue

            kpts = hint_keypoints_to_yolo(
                kpts_field,
                ann.get("existence"),
                ann.get("occlusion"),
                w,
                h,
            )

            dst_jpg = out_images / jpg.name
            dst_txt = out_labels / f"{stem}.txt"

            if dst_jpg.exists() or dst_jpg.is_symlink():
                dst_jpg.unlink()
            if link_mode == "symlink":
                dst_jpg.symlink_to(jpg.resolve())
            elif link_mode == "hardlink":
                os.link(jpg, dst_jpg)
            elif link_mode == "copy":
                from shutil import copyfile
                copyfile(jpg, dst_jpg)
            else:
                raise ValueError(link_mode)

            cx, cy, bw, bh = yolo_box
            kpt_str = " ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in kpts)
            dst_txt.write_text(
                f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpt_str}\n"
            )

            stats.n_images += 1
            stats.n_labels += 1

    return stats


def write_data_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    # Ultralytics resolves relative `path` against the project, so use absolute.
    # kpt_shape: 21 hand keypoints (MANO order) x [x, y, v]; required by the
    # WiLoR detector.pt pose head.  No flip_idx because train.py runs fliplr=0
    # (left/right hand classes would be mislabeled by a horizontal flip).
    content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"kpt_shape: [{N_KEYPOINTS}, 3]\n"
        f"names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    yaml_path.write_text(content)
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint-root", type=Path, default=DEFAULT_HINT_ROOT)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "hint_yolo",
        help="output dataset directory (default: ./hint_yolo next to this script)",
    )
    ap.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="how images are placed under out_dir/images (symlink is safe and cheap)",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, Stats] = {}
    for split, subdirs in SPLITS.items():
        src_dirs = [args.hint_root / s for s in subdirs]
        s = process_split(
            src_dirs,
            out_dir / "images" / split,
            out_dir / "labels" / split,
            link_mode=args.link_mode,
        )
        totals[split] = s
        print(
            f"[{split}] images={s.n_images} labels={s.n_labels} "
            f"skipped_no_json={s.n_skipped_no_json} skipped_bad_bbox={s.n_skipped_bad_bbox}"
        )

    yaml_path = write_data_yaml(out_dir)
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
