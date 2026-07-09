"""Rotation utilities: axis-angle -> rotation matrix / Euler (roll, pitch, yaw)."""

from __future__ import annotations

import cv2
import numpy as np


# Parent of each of the 15 MANO hand_pose joints within the pose chain
# (-1 = wrist / global_orient root).
# Joint order: index(0-2), middle(3-5), pinky(6-8), ring(9-11), thumb(12-14).
MANO_PARENTS = [-1, 0, 1, -1, 3, 4, -1, 6, 7, -1, 9, 10, -1, 12, 13]

# Mapping from the 15 MANO pose joints to the 21-keypoint indices
# (wrist=0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20).
MANO_TO_KEYPOINT = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]

# Distal MANO joints whose rigid phalanx carries each fingertip.
# Finger order: index, middle, pinky, ring, thumb (matches MANO joint order).
FINGERTIP_PARENT_JOINTS = [2, 5, 8, 11, 14]

# 21-keypoint indices of the fingertips, same finger order as above.
FINGERTIP_KEYPOINTS = [8, 12, 20, 16, 4]

FINGER_NAMES = ["index", "middle", "pinky", "ring", "thumb"]


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle vectors to (..., 3, 3) rotation matrices."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    flat = rotvec.reshape(-1, 3)
    mats = np.stack([cv2.Rodrigues(v)[0] for v in flat])
    return mats.reshape(*rotvec.shape[:-1], 3, 3)


def matrix_to_euler(rotmat: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrices to (..., 3) Euler angles.

    Uses the intrinsic Z-Y-X (yaw-pitch-roll) convention and returns angles
    ordered as (roll, pitch, yaw) about the (x, y, z) axes.
    """
    rotmat = np.asarray(rotmat, dtype=np.float64)
    flat = rotmat.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 3))
    for i, m in enumerate(flat):
        sp = np.clip(-m[2, 0], -1.0, 1.0)
        pitch = np.arcsin(sp)
        if abs(sp) < 1.0 - 1e-8:
            roll = np.arctan2(m[2, 1], m[2, 2])
            yaw = np.arctan2(m[1, 0], m[0, 0])
        else:  # gimbal lock: yaw is unrecoverable, fold it into roll
            roll = np.arctan2(-m[1, 2], m[1, 1])
            yaw = 0.0
        out[i] = (roll, pitch, yaw)
    if degrees:
        out = np.degrees(out)
    return out.reshape(*rotmat.shape[:-2], 3)


def axis_angle_to_euler(rotvec: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Convert (..., 3) axis-angle vectors to (roll, pitch, yaw) Euler angles."""
    return matrix_to_euler(axis_angle_to_matrix(rotvec), degrees=degrees)


def cumulative_joint_rotations(
    global_orient: np.ndarray, hand_pose: np.ndarray
) -> np.ndarray:
    """Compose per-joint relative rotations along the MANO kinematic chain.

    Parameters
    ----------
    global_orient : (3,) axis-angle of the wrist in camera space.
    hand_pose : (15, 3) per-joint axis-angle relative to each parent joint.

    Returns
    -------
    (15, 3, 3) camera-space orientation of each hand_pose joint.
    """
    root = axis_angle_to_matrix(np.reshape(global_orient, 3))
    local = axis_angle_to_matrix(np.reshape(hand_pose, (15, 3)))
    out = np.empty_like(local)
    for j, p in enumerate(MANO_PARENTS):
        parent = root if p < 0 else out[p]
        out[j] = parent @ local[j]
    return out


def fingertip_rotations(
    global_orient: np.ndarray, hand_pose: np.ndarray
) -> np.ndarray:
    """Camera-space orientation of each fingertip.

    MANO has no pose parameter for the tips: each tip is the end of the
    rigid distal phalanx, so its orientation equals the cumulative rotation
    of the distal joint.

    Returns
    -------
    (5, 3, 3) rotation matrices in (index, middle, pinky, ring, thumb)
    order; the corresponding 21-keypoint indices are
    :data:`FINGERTIP_KEYPOINTS`.
    """
    return cumulative_joint_rotations(global_orient, hand_pose)[
        FINGERTIP_PARENT_JOINTS
    ]
