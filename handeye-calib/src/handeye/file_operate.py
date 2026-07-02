#!/usr/bin/env python3
# coding: utf-8
import csv
import numpy as np
import os
import sys
import math

try:
    import transforms3d as tfs
except Exception:
    tfs = None


def _quat_xyzw_to_euler_deg(qx: float, qy: float, qz: float, qw: float):
    if tfs is None:
        raise ImportError("transforms3d is required to convert quaternion CSV to Euler angles")
    rx, ry, rz = tfs.euler.quat2euler((qw, qx, qy, qz))
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def _parse_numeric_fields(line):
    nums = []
    for d in line[1:]:
        # Keep backward-compat behavior: ignore the leading label; parse the rest as floats
        # (CSV may include whitespace)
        s = str(d).strip()
        if not s:
            continue
        # Preserve original intent: skip pure alphabetic tokens
        if s.isalpha():
            continue
        nums.append(float(s))
    return nums

def read_handeye_data(path):
    hand = []
    eye = []
    reader = csv.reader(open(path, "r"))
    for line in reader:
        if not line:
            continue
        tag = str(line[0]).strip().lower()
        if tag not in ("hand", "eye"):
            continue

        nums = _parse_numeric_fields(line)

        # Supported formats per row:
        # - 6 numbers:  x, y, z, rx, ry, rz  (Euler angles in degrees)
        # - 7 numbers:  x, y, z, qx, qy, qz, qw  (Quaternion in ROS order)
        if len(nums) == 6:
            out = nums
        elif len(nums) == 7:
            x, y, z, qx, qy, qz, qw = nums
            rx, ry, rz = _quat_xyzw_to_euler_deg(qx, qy, qz, qw)
            out = [x, y, z, rx, ry, rz]
        else:
            raise ValueError(f"Invalid {tag} row: expected 6 (euler) or 7 (quat) numbers, got {len(nums)}: {line}")

        if tag == "hand":
            hand.extend(out)
        else:
            eye.extend(out)
    return np.asarray(hand, dtype=float, order=None), np.asarray(eye, dtype=float, order=None)

def save_file(path,data):
    if str(path).startswith("~"):
        path = path.replace("~",str(os.getenv("HOME")))

    if  not os.path.exists(path[:path.rfind("/")]):
        os.mkdir(path[:path.rfind("/")])

    with open(path,'w') as wf:
        wf.write(str(data))
        wf.close()
        

    
