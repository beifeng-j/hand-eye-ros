#!/usr/bin/env python3
import argparse
import csv
import math
import re
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


HAND_EULER_ORDER = "XYZ"
EYE_EULER_ORDER = "xyz"


def make_transform(values: Sequence[float], euler_order: str) -> np.ndarray:
    x, y, z, rx, ry, rz = values
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler(euler_order, [rx, ry, rz], degrees=True).as_matrix()
    transform[:3, 3] = [x, y, z]
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def read_hand_eye_poses(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    hands: List[np.ndarray] = []
    eyes: List[np.ndarray] = []

    with open(path, "r", newline="") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = [part for part in re.split(r"[\s,]+", line) if part]
            tag = parts[0].lower()
            if tag not in ("hand", "eye"):
                continue
            if len(parts) != 7:
                raise ValueError(
                    f"line {line_no}: expected '{tag},x,y,z,rx,ry,rz', got {len(parts) - 1} values"
                )

            values = [float(value) for value in parts[1:]]
            if tag == "hand":
                hands.append(make_transform(values, HAND_EULER_ORDER))
            else:
                eyes.append(make_transform(values, EYE_EULER_ORDER))

    if len(hands) != len(eyes):
        raise ValueError(f"hand rows ({len(hands)}) and eye rows ({len(eyes)}) are not equal")
    if len(hands) < 3:
        raise ValueError("at least 3 hand/eye pose pairs are required")

    return hands, eyes


def build_motion_pairs(
    hand_poses: Sequence[np.ndarray],
    eye_poses: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    motion_a: List[np.ndarray] = []
    motion_b: List[np.ndarray] = []
    sample_count = len(hand_poses)

    # Adjacent pairs plus wider-baseline pairs keep the solution stable without
    # over-weighting many nearly duplicate pair combinations.
    pair_indices = [(i, i + 1) for i in range(sample_count - 1)]
    pair_indices += [(i, i + 5) for i in range(sample_count - 5)]
    pair_indices += [(i, i + 10) for i in range(sample_count - 10)]

    for i, j in pair_indices:
        motion_a.append(invert_transform(hand_poses[j]) @ hand_poses[i])
        motion_b.append(eye_poses[j] @ invert_transform(eye_poses[i]))

    return motion_a, motion_b


def solve_eye_in_hand(hand_poses: Sequence[np.ndarray], eye_poses: Sequence[np.ndarray]) -> np.ndarray:
    motion_a, motion_b = build_motion_pairs(hand_poses, eye_poses)

    alpha = []
    beta = []
    valid_motion_a = []
    valid_motion_b = []

    for a, b in zip(motion_a, motion_b):
        alpha_vec = R.from_matrix(a[:3, :3]).as_rotvec()
        beta_vec = R.from_matrix(b[:3, :3]).as_rotvec()
        if np.linalg.norm(alpha_vec) > 1e-5 and np.linalg.norm(beta_vec) > 1e-5:
            alpha.append(alpha_vec)
            beta.append(beta_vec)
            valid_motion_a.append(a)
            valid_motion_b.append(b)

    if len(alpha) < 2:
        raise ValueError("not enough rotational motion to solve hand-eye calibration")

    rotation_solution, _ = R.align_vectors(np.asarray(alpha), np.asarray(beta))
    r_end_camera = rotation_solution.as_matrix()

    lhs_blocks = []
    rhs_blocks = []
    for a, b in zip(valid_motion_a, valid_motion_b):
        lhs_blocks.append(a[:3, :3] - np.eye(3))
        rhs_blocks.append(r_end_camera @ b[:3, 3] - a[:3, 3])

    t_end_camera = np.linalg.lstsq(
        np.vstack(lhs_blocks),
        np.concatenate(rhs_blocks),
        rcond=None,
    )[0]

    end_camera = np.eye(4, dtype=np.float64)
    end_camera[:3, :3] = r_end_camera
    end_camera[:3, 3] = t_end_camera
    return end_camera


def consistency_metrics(
    hand_poses: Sequence[np.ndarray],
    eye_poses: Sequence[np.ndarray],
    end_camera: np.ndarray,
) -> Tuple[float, float, float, float]:
    base_marker_poses = [hand @ end_camera @ eye for hand, eye in zip(hand_poses, eye_poses)]
    translations = np.asarray([pose[:3, 3] for pose in base_marker_poses])
    translation_mean = translations.mean(axis=0)
    translation_errors = np.linalg.norm(translations - translation_mean, axis=1)

    rotations = R.from_matrix(np.asarray([pose[:3, :3] for pose in base_marker_poses]))
    rotation_mean = rotations.mean()
    rotation_errors = np.asarray(
        [math.degrees((rotation_mean.inv() * rotation).magnitude()) for rotation in rotations]
    )

    return (
        float(np.sqrt(np.mean(translation_errors**2))),
        float(np.max(translation_errors)),
        float(np.sqrt(np.mean(rotation_errors**2))),
        float(np.max(rotation_errors)),
    )


def print_result(end_camera: np.ndarray, metrics: Tuple[float, float, float, float]) -> None:
    np.set_printoptions(precision=8, suppress=True)

    rotation = R.from_matrix(end_camera[:3, :3])
    euler_xyz = rotation.as_euler("xyz", degrees=True)
    quat_xyzw = rotation.as_quat()

    print("=== Eye-in-hand calibration ===")
    print(f"hand Euler order: {HAND_EULER_ORDER}")
    print(f"eye Euler order:  {EYE_EULER_ORDER}")
    print("transform: end_link -> camera")
    print()
    print("4x4 matrix end_link_T_camera:")
    print(end_camera)
    print()
    print(
        "translation xyz (m): "
        f"{end_camera[0, 3]:.8f}, {end_camera[1, 3]:.8f}, {end_camera[2, 3]:.8f}"
    )
    print(f"euler xyz (deg):    {euler_xyz[0]:.8f}, {euler_xyz[1]:.8f}, {euler_xyz[2]:.8f}")
    print(f"quaternion xyzw:    {quat_xyzw[0]:.8f}, {quat_xyzw[1]:.8f}, {quat_xyzw[2]:.8f}, {quat_xyzw[3]:.8f}")
    print()
    print("Consistency check: base_link -> marker should be stable")
    print(f"translation RMS/max: {metrics[0]:.8f} m / {metrics[1]:.8f} m")
    print(f"rotation RMS/max:    {metrics[2]:.8f} deg / {metrics[3]:.8f} deg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute eye-in-hand calibration for hand=XYZ Euler and eye=xyz Euler 6D CSV data."
    )
    parser.add_argument(
        "csv_path",
        help="Input file. Each row: hand x y z rx ry rz or eye x y z rx ry rz. Commas, tabs, and spaces are supported.",
    )
    args = parser.parse_args()

    hand_poses, eye_poses = read_hand_eye_poses(args.csv_path)
    end_camera = solve_eye_in_hand(hand_poses, eye_poses)
    metrics = consistency_metrics(hand_poses, eye_poses, end_camera)
    print_result(end_camera, metrics)


if __name__ == "__main__":
    main()
