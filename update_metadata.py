"""
Update metadata files in episode/run folders with calibrated intrinsic values
from calibration.npz.

Supports two formats:
  - Old format: folder with metadata.npz  (e.g. data/run_6_high_accuracy)
  - New format: zarr episode folder with metadata.json  (e.g. data/episode_*.zarr)

Usage:
    python update_metadata.py <folder_path>       # single folder (auto-detects format)
    python update_metadata.py --all <data_dir>    # all episode/run folders under data_dir

Examples:
    python update_metadata.py data/run_6_high_accuracy
    python update_metadata.py data/episode_20260507_232139.zarr
    python update_metadata.py --all data/
"""

import numpy as np
import os
import sys
import json


def get_intrinsic_dict(camera_matrix, dist_coeffs, width, height):
    return {
        'fx':     float(camera_matrix[0, 0]),
        'fy':     float(camera_matrix[1, 1]),
        'ppx':    float(camera_matrix[0, 2]),
        'ppy':    float(camera_matrix[1, 2]),
        'width':  int(width),
        'height': int(height),
        'model':  'Brown-Conrady',
        'coeffs': dist_coeffs.flatten().tolist(),
    }


def load_calibration(calib_path="calibration.npz"):
    if not os.path.exists(calib_path):
        print(f"ERROR: {calib_path} not found in current directory")
        sys.exit(1)
    calib = np.load(calib_path, allow_pickle=True)
    return calib['camera_matrix'], calib['dist_coeffs'], float(calib['rms_error'])


def print_calibration(camera_matrix, dist_coeffs, rms_error):
    print(f"Calibration: rms={rms_error:.6f} px")
    print(f"  fx={camera_matrix[0,0]:.4f}  fy={camera_matrix[1,1]:.4f}  "
          f"cx={camera_matrix[0,2]:.4f}  cy={camera_matrix[1,2]:.4f}")
    print(f"  dist={dist_coeffs.flatten().tolist()}\n")


# ── New format (metadata.json) ─────────────────────────────────────────────

def update_json(folder_path, camera_matrix, dist_coeffs, rms_error):
    metadata_path = os.path.join(folder_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    h, w = meta["image_resolution"]   # stored as [H, W]
    intr = get_intrinsic_dict(camera_matrix, dist_coeffs, w, h)

    meta["depth_intrinsics"]      = intr
    meta["color_intrinsics"]      = intr
    meta["calibration_rms_error"] = rms_error

    with open(metadata_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  [new] {metadata_path}")
    print(f"    fx={intr['fx']:.2f}  fy={intr['fy']:.2f}  "
          f"ppx={intr['ppx']:.2f}  ppy={intr['ppy']:.2f}  rms={rms_error:.6f}")

    # Verify round-trip
    with open(metadata_path, 'r') as f:
        check = json.load(f)
    assert abs(check["depth_intrinsics"]["fx"] - intr["fx"]) < 1e-6
    print(f"    verified OK")


# ── Old format (metadata.npz) ──────────────────────────────────────────────

def update_npz(folder_path, camera_matrix, dist_coeffs, rms_error):
    metadata_path = os.path.join(folder_path, "metadata.npz")
    meta = np.load(metadata_path, allow_pickle=True)

    color_shape = meta['color_shape']
    width  = int(color_shape[2])   # (N, H, W, 3)
    height = int(color_shape[1])
    intr   = get_intrinsic_dict(camera_matrix, dist_coeffs, width, height)

    np.savez(metadata_path,
        frame_count              = meta['frame_count'],
        depth_scale              = meta['depth_scale'],
        timestamps               = meta['timestamps'],
        depth_intrinsics         = intr,
        color_intrinsics         = intr,
        depth_to_color_extrinsics= meta['depth_to_color_extrinsics'],
        depth_shape              = meta['depth_shape'],
        color_shape              = meta['color_shape'],
        color_format             = meta['color_format'],
        recording_time           = meta['recording_time'],
        calibration_rms_error    = rms_error,
    )

    print(f"  [old] {metadata_path}")
    print(f"    fx={intr['fx']:.2f}  fy={intr['fy']:.2f}  "
          f"ppx={intr['ppx']:.2f}  ppy={intr['ppy']:.2f}  rms={rms_error:.6f}")

    verify = np.load(metadata_path, allow_pickle=True)
    assert abs(float(verify['depth_intrinsics'].item()['fx']) - intr['fx']) < 1e-6
    print(f"    verified OK")


# ── Dispatch ───────────────────────────────────────────────────────────────

def process_folder(folder_path, camera_matrix, dist_coeffs, rms_error):
    folder_path = folder_path.rstrip("/\\")
    json_path = os.path.join(folder_path, "metadata.json")
    npz_path  = os.path.join(folder_path, "metadata.npz")

    if os.path.exists(json_path):
        update_json(folder_path, camera_matrix, dist_coeffs, rms_error)
    elif os.path.exists(npz_path):
        update_npz(folder_path, camera_matrix, dist_coeffs, rms_error)
    else:
        print(f"  SKIP (no metadata.json or metadata.npz): {folder_path}")


def find_episode_folders(data_dir):
    folders = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        if (os.path.exists(os.path.join(full, "metadata.json")) or
                os.path.exists(os.path.join(full, "metadata.npz"))):
            folders.append(full)
    return folders


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    camera_matrix, dist_coeffs, rms_error = load_calibration("calibration.npz")
    print_calibration(camera_matrix, dist_coeffs, rms_error)

    if sys.argv[1] == "--all":
        if len(sys.argv) < 3:
            print("Usage: python update_metadata.py --all <data_dir>")
            sys.exit(1)
        data_dir = sys.argv[2]
        folders = find_episode_folders(data_dir)
        if not folders:
            print(f"No episode folders found under {data_dir}")
            sys.exit(1)
        print(f"Found {len(folders)} folder(s) under {data_dir}:\n")
        for folder in folders:
            process_folder(folder, camera_matrix, dist_coeffs, rms_error)
    else:
        process_folder(sys.argv[1], camera_matrix, dist_coeffs, rms_error)


if __name__ == "__main__":
    main()
