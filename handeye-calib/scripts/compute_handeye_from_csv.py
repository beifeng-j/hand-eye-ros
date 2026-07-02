#!/usr/bin/env python3
import os
import csv
import math
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
import glob


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    # Normalize to be safe
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        raise ValueError('Zero-norm quaternion')
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)
    return R


def make_h(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3:4] = t.reshape(3, 1)
    return H


def rot_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    Rt = R1.T @ R2
    tr = np.trace(Rt)
    cos_th = max(min((tr - 1.0) * 0.5, 1.0), -1.0)
    return abs(math.degrees(math.acos(cos_th)))


def evaluate_consistency(Hg_list: List[np.ndarray], Hc_list: List[np.ndarray], Hcg: np.ndarray) -> Tuple[float, float, float, float]:
    Hbt = [Hg @ Hcg @ Hc for Hg, Hc in zip(Hg_list, Hc_list)]
    R_ref, t_ref = Hbt[0][:3, :3], Hbt[0][:3, 3]
    rot_errs, trans_errs = [], []
    for H in Hbt:
        R, t = H[:3, :3], H[:3, 3]
        rot_errs.append(rot_angle_deg(R_ref, R))
        trans_errs.append(np.linalg.norm(t_ref - t))
    rot_rms = math.sqrt(sum(e*e for e in rot_errs) / len(rot_errs))
    trans_rms = math.sqrt(sum(e*e for e in trans_errs) / len(trans_errs))
    return rot_rms, max(rot_errs), trans_rms, max(trans_errs)


def evaluate_ax_eq_xb(Hg_list: List[np.ndarray], Hc_list: List[np.ndarray], Hcg: np.ndarray) -> Tuple[float, float, float]:
    """Pairwise AX=XB residuals across all pairs (i<j).
    Returns (rot_rms_deg, trans_rms_m, frob_rms)."""
    N = len(Hg_list)
    if N < 2:
        return 0.0, 0.0, 0.0
    rot_sq = 0.0; trans_sq = 0.0; frob_sq = 0.0; cnt = 0
    I3 = np.eye(3)
    for i in range(N):
        for j in range(i+1, N):
            A = np.linalg.inv(Hg_list[j]) @ Hg_list[i]
            B = Hc_list[j] @ np.linalg.inv(Hc_list[i])
            Delta = (A @ Hcg) @ np.linalg.inv(Hcg @ B)
            R_d = Delta[:3, :3]
            t_d = Delta[:3, 3]
            rot_sq += rot_angle_deg(I3, R_d)**2
            trans_sq += float(np.linalg.norm(t_d)**2)
            frob_sq += float(np.linalg.norm(Delta, ord='fro')**2)
            cnt += 1
    return math.sqrt(rot_sq/cnt), math.sqrt(trans_sq/cnt), math.sqrt(frob_sq/cnt)


def invert_rt_lists(R_list: List[np.ndarray], t_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    R_inv, t_inv = [], []
    for R, t in zip(R_list, t_list):
        Rt = R.T
        ti = -Rt @ t
        R_inv.append(Rt)
        t_inv.append(ti)
    return R_inv, t_inv


def compute_once(R_g2b: List[np.ndarray], t_g2b: List[np.ndarray],
                 R_t2c: List[np.ndarray], t_t2c: List[np.ndarray],
                 Hg_list: List[np.ndarray], Hc_list: List[np.ndarray],
                 method_code: int) -> Dict[str, Any]:
    R_cg, t_cg = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method_code)
    Hcg = make_h(R_cg, t_cg)
    rot_rms, rot_max, trans_rms, trans_max = evaluate_consistency(Hg_list, Hc_list, Hcg)
    ax_rot, ax_trans, ax_frob = evaluate_ax_eq_xb(Hg_list, Hc_list, Hcg)
    score = trans_rms + 0.01 * rot_rms  # prioritize translation (m) while considering rotation (deg)
    return {
        'Hcg': Hcg,
        'rot_rms': rot_rms, 'rot_max': rot_max, 'trans_rms': trans_rms, 'trans_max': trans_max,
        'ax_rot': ax_rot, 'ax_trans': ax_trans, 'ax_frob': ax_frob,
        'score': score,
    }


def compute_from_csv(csv_path: str, method: str = 'tsai', invert_hand: bool = False, invert_eye: bool = False, grid: bool = False):
    methods = {
        'tsai': cv2.CALIB_HAND_EYE_TSAI,
        'park': cv2.CALIB_HAND_EYE_PARK,
        'horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if method not in methods:
        raise ValueError(f'Unknown method {method}, choose from {list(methods)}')

    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if len(rows) < 3:
        raise RuntimeError('需要至少3组数据')

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []
    Hg_list, Hc_list = [], []

    for r in rows:
        # End-effector in base (Base<-Gripper)
        ex = float(r['end_ox']); ey = float(r['end_oy']); ez = float(r['end_oz']); ew = float(r['end_ow'])
        et = np.array([float(r['end_px']), float(r['end_py']), float(r['end_pz'])], dtype=np.float64)
        Rbg = quat_to_R(ex, ey, ez, ew)
        Hg = make_h(Rbg, et)

        # Target in camera (Camera<-Target)
        ax = float(r['aruco_ox']); ay = float(r['aruco_oy']); az = float(r['aruco_oz']); aw = float(r['aruco_ow'])
        at = np.array([float(r['aruco_px']), float(r['aruco_py']), float(r['aruco_pz'])], dtype=np.float64)
        Rct = quat_to_R(ax, ay, az, aw)
        Hc = make_h(Rct, at)

        R_g2b.append(Rbg)
        t_g2b.append(et.reshape(3, 1))
        R_t2c.append(Rct)
        t_t2c.append(at.reshape(3, 1))
        Hg_list.append(Hg)
        Hc_list.append(Hc)

    # Prepare inverted options
    def prepare_variant(inv_hand: bool, inv_eye: bool):
        Rg, tg = (R_g2b, t_g2b)
        Rc, tc = (R_t2c, t_t2c)
        HgL, HcL = (Hg_list, Hc_list)
        if inv_hand:
            Rg, tg = invert_rt_lists(Rg, tg)
            HgL = [np.linalg.inv(H) for H in HgL]
        if inv_eye:
            Rc, tc = invert_rt_lists(Rc, tc)
            HcL = [np.linalg.inv(H) for H in HcL]
        return Rg, tg, Rc, tc, HgL, HcL

    np.set_printoptions(precision=8, suppress=True)
    if grid:
        print('=== Grid Search over methods and inversion flags ===')
        results = []
        for m_name, m_code in methods.items():
            for ih in [False, True]:
                for ie in [False, True]:
                    Rg, tg, Rc, tc, HgL, HcL = prepare_variant(ih, ie)
                    try:
                        res = compute_once(Rg, tg, Rc, tc, HgL, HcL, m_code)
                        res.update({'method': m_name, 'invert_hand': ih, 'invert_eye': ie})
                        results.append(res)
                    except cv2.error as e:
                        continue
        if not results:
            raise RuntimeError('No valid solution in grid search')
        # Sort by score
        results.sort(key=lambda r: r['score'])
        best = results[0]
        print('Top-3 solutions (by score = trans_rms + 0.01*rot_rms):')
        for r in results[:3]:
            print(f"- {r['method']}, invert_hand={r['invert_hand']}, invert_eye={r['invert_eye']} | "
                  f"trans_rms={r['trans_rms']:.6f} m, rot_rms={r['rot_rms']:.3f} deg | "
                  f"AX=XB rot={r['ax_rot']:.3f} deg, trans={r['ax_trans']:.4f} m")
        Hcg = best['Hcg']
        print('\n=== Best Solution ===')
        print(f"Method: {best['method']}, invert_hand={best['invert_hand']}, invert_eye={best['invert_eye']}")
        print('Hcg (Camera<-Gripper):\n', Hcg)
        print('\nRMS consistency (Base<-Target across samples):')
        print(f"  Rotation RMS: {best['rot_rms']:.4f} deg  (max {best['rot_max']:.4f})")
        print(f"  Translation RMS: {best['trans_rms']:.6f} m  (max {best['trans_max']:.6f})")
        print('\n[AX=XB] residuals:')
        print(f"  Rotation RMS: {best['ax_rot']:.4f} deg")
        print(f"  Translation RMS: {best['ax_trans']:.6f} m")
        print(f"  Frobenius RMS: {best['ax_frob']:.6f}")
        Hgc = np.linalg.inv(Hcg)
        print('\nHgc (Gripper<-Camera):\n', Hgc)
    else:
        # Single run with provided flags
        Rg, tg, Rc, tc, HgL, HcL = prepare_variant(invert_hand, invert_eye)
        res = compute_once(Rg, tg, Rc, tc, HgL, HcL, methods[method])
        Hcg = res['Hcg']
        print('=== Hand-Eye (Eye-in-Hand) ===')
        print(f'Method: {method}, invert_hand={invert_hand}, invert_eye={invert_eye}')
        print('Hcg (Camera<-Gripper):\n', Hcg)
        print('\nRMS consistency (Base<-Target across samples):')
        print(f"  Rotation RMS: {res['rot_rms']:.4f} deg  (max {res['rot_max']:.4f})")
        print(f"  Translation RMS: {res['trans_rms']:.6f} m  (max {res['trans_max']:.6f})")
        print('\n[AX=XB] residuals:')
        print(f"  Rotation RMS: {res['ax_rot']:.4f} deg")
        print(f"  Translation RMS: {res['ax_trans']:.6f} m")
        print(f"  Frobenius RMS: {res['ax_frob']:.6f}")
        Hgc = np.linalg.inv(Hcg)
        print('\nHgc (Gripper<-Camera):\n', Hgc)


def main():
    parser = argparse.ArgumentParser(description='Compute eye-in-hand hand-eye calibration from CSV collected snapshots')
    parser.add_argument('--csv', default=None, help='CSV file path; if omitted, auto-pick latest logs/pose_pairs_*.csv')
    parser.add_argument('--method', default='tsai', choices=['tsai','park','horaud','daniilidis'])
    parser.add_argument('--invert_hand', action='store_true', help='Invert end-effector pose if data is Gripper<-Base instead of Base<-Gripper')
    parser.add_argument('--invert_eye', action='store_true', help='Invert aruco pose if data is Target<-Camera instead of Camera<-Target')
    parser.add_argument('--grid', action='store_true', help='Grid search methods and inversion flags; print best result')
    args = parser.parse_args()

    # If no CSV provided, try to auto-detect the latest one under common logs paths
    csv_path: Optional[str] = args.csv
    if not csv_path:
        candidates = []
        # Current working directory logs
        candidates += glob.glob(os.path.join(os.getcwd(), 'logs', 'pose_pairs_*.csv'))
        # Workspace logs (typical for this project)
        candidates += glob.glob('/home/ds/g1/songling_ws/logs/pose_pairs_*.csv')
        # Sort by mtime descending
        candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
        if candidates:
            csv_path = candidates[0]
            print(f'[info] 未提供 --csv，自动选择最新文件: {csv_path}')
        else:
            # As a last resort, you can hardcode your path here
            # csv_path = '/home/ds/g1/songling_ws/logs/pose_pairs_20251016_153612.csv'
            raise SystemExit('找不到日志 CSV。请用 --csv 指定，或将文件放在 logs/pose_pairs_*.csv')

    compute_from_csv(csv_path, args.method, invert_hand=args.invert_hand, invert_eye=args.invert_eye, grid=args.grid)


if __name__ == '__main__':
    main()
