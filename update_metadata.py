"""
Update metadata.npz in a folder with calibrated intrinsic values from calibration.npz.

Usage:
    python update_metadata.py <folder_path>
    
Example:
    python update_metadata.py data/run_6_high_accuracy

This will:
1. Read calibration.npz to get the calibrated camera matrix and distortion coefficients
2. Extract intrinsic parameters (fx, fy, ppx, ppy) from the calibration
3. Update both depth_intrinsics and color_intrinsics in metadata.npz with the calibrated values
4. Preserve other metadata fields and image dimensions
"""

import numpy as np
import os
import sys
import json


def get_intrinsic_dict(camera_matrix, dist_coeffs, width, height):
    """Convert from calibration format to intrinsic dictionary format."""
    return {
        'fx': float(camera_matrix[0, 0]),
        'fy': float(camera_matrix[1, 1]),
        'ppx': float(camera_matrix[0, 2]),
        'ppy': float(camera_matrix[1, 2]),
        'width': int(width),
        'height': int(height),
        'model': 'Brown-Conrady',  # Standard distortion model
        'coeffs': dist_coeffs.flatten().tolist(),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_metadata.py <folder_path>")
        print("Example: python update_metadata.py data/run_6_high_accuracy")
        sys.exit(1)

    folder_path = sys.argv[1]
    metadata_path = os.path.join(folder_path, "metadata.npz")
    calib_path = "calibration.npz"

    # Validate files exist
    if not os.path.exists(calib_path):
        print(f"ERROR: calibration.npz not found in current directory")
        sys.exit(1)

    if not os.path.exists(metadata_path):
        print(f"ERROR: metadata.npz not found at {metadata_path}")
        sys.exit(1)

    print(f"Loading calibration from: {calib_path}")
    calib = np.load(calib_path, allow_pickle=True)
    camera_matrix = calib['camera_matrix']
    dist_coeffs = calib['dist_coeffs']
    rms_error = float(calib['rms_error'])

    print(f"Loading metadata from: {metadata_path}")
    metadata = np.load(metadata_path, allow_pickle=True)

    # Extract existing metadata fields
    frame_count = metadata['frame_count']
    depth_scale = metadata['depth_scale']
    timestamps = metadata['timestamps']
    depth_shape = metadata['depth_shape']
    color_shape = metadata['color_shape']
    color_format = metadata['color_format']
    recording_time = metadata['recording_time']
    depth_to_color_extrinsics = metadata['depth_to_color_extrinsics']

    # Get image dimensions from existing metadata
    width = int(color_shape[2])  # color_shape is (N, H, W, 3)
    height = int(color_shape[1])

    # Create new intrinsic dicts with calibrated values
    new_intrinsics = get_intrinsic_dict(camera_matrix, dist_coeffs, width, height)

    print("\n" + "="*60)
    print("Calibration parameters:")
    print("="*60)
    print(f"RMS error: {rms_error:.6f} px")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}")
    print(f"Image size: {width}x{height}")

    print("\n" + "="*60)
    print("New intrinsics (will replace both depth and color):")
    print("="*60)
    for key, val in new_intrinsics.items():
        if key == 'coeffs':
            print(f"  {key}: {[f'{v:.6f}' for v in val]}")
        else:
            print(f"  {key}: {val}")

    # Prepare updated metadata dictionary
    updated_metadata = {
        'frame_count': frame_count,
        'depth_scale': depth_scale,
        'timestamps': timestamps,
        'depth_intrinsics': new_intrinsics,
        'color_intrinsics': new_intrinsics,  # Same intrinsics for both
        'depth_to_color_extrinsics': depth_to_color_extrinsics,
        'depth_shape': depth_shape,
        'color_shape': color_shape,
        'color_format': color_format,
        'recording_time': recording_time,
        'calibration_rms_error': rms_error,
    }

    # Save back to metadata.npz (in-place update)
    print(f"\nSaving updated metadata to: {metadata_path}")
    np.savez(metadata_path, **updated_metadata)

    print("✓ Metadata updated successfully!")
    print("\nVerification:")
    # Reload and verify
    verify = np.load(metadata_path, allow_pickle=True)
    verify_depth_intr = verify['depth_intrinsics'].item()
    verify_color_intr = verify['color_intrinsics'].item()
    print(f"  depth_intrinsics fx: {verify_depth_intr['fx']:.1f}")
    print(f"  color_intrinsics fx: {verify_color_intr['fx']:.1f}")
    print(f"  calibration_rms_error: {float(verify['calibration_rms_error']):.6f}")


if __name__ == "__main__":
    main()
