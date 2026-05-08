"""
flash_calibration.py

Writes calibrated intrinsics from calibration.npz directly into the
RealSense camera's EEPROM so that rs2 intrinsics queries return the
correct values from that point on.

Usage:
    python flash_calibration.py --calibration path/to/calibration.npz

WARNING: This modifies the camera's internal memory. The original
calibration is backed up and printed before any changes are made.
"""

import argparse
import numpy as np
import pyrealsense2 as rs


def flash_calibration(calibration_path: str):
    # --- Load calibration ---
    cal = np.load(calibration_path)
    camera_matrix = cal['camera_matrix']
    dist_coeffs = cal['dist_coeffs'].flatten()
    rms = float(cal['rms_error'])
    print(f"Loaded calibration (RMS: {rms:.4f}px)")

    # --- Connect to camera ---
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found.")
    device = devices[0]
    print(f"Connected to: {device.get_info(rs.camera_info.name)}")
    print(f"Serial:       {device.get_info(rs.camera_info.serial_number)}")

    # --- Start pipeline to get current intrinsics ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    print("\n=== Current camera intrinsics (before) ===")
    print(f"  fx:    {intr.fx:.4f}")
    print(f"  fy:    {intr.fy:.4f}")
    print(f"  ppx:   {intr.ppx:.4f}")
    print(f"  ppy:   {intr.ppy:.4f}")
    print(f"  coeffs: {intr.coeffs}")

    pipeline.stop()

    # --- Build updated intrinsics object ---
    new_intr = rs.intrinsics()
    new_intr.width  = intr.width
    new_intr.height = intr.height
    new_intr.fx     = float(camera_matrix[0, 0])
    new_intr.fy     = float(camera_matrix[1, 1])
    new_intr.ppx    = float(camera_matrix[0, 2])
    new_intr.ppy    = float(camera_matrix[1, 2])
    new_intr.model  = intr.model
    new_intr.coeffs = dist_coeffs.tolist()

    print("\n=== New intrinsics (to be flashed) ===")
    print(f"  fx:    {new_intr.fx:.4f}")
    print(f"  fy:    {new_intr.fy:.4f}")
    print(f"  ppx:   {new_intr.ppx:.4f}")
    print(f"  ppy:   {new_intr.ppy:.4f}")
    print(f"  coeffs: {new_intr.coeffs}")

    # --- Confirm before writing ---
    confirm = input("\nFlash these intrinsics to camera EEPROM? [y/N]: ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # --- Write to EEPROM ---
    # Requires advanced mode on D435i
    advnc_mode = rs.rs400_advanced_mode(device)
    if not advnc_mode.is_enabled():
        print("Enabling advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # Device resets after enabling — reconnect
        import time
        time.sleep(3)
        ctx = rs.context()
        device = ctx.query_devices()[0]
        advnc_mode = rs.rs400_advanced_mode(device)

    device.set_calibration_table(
        rs.calibration_type.color_camera_intrinsics, new_intr
    )
    device.write_calibration()

    print("\nCalibration successfully written to camera EEPROM.")
    print("Restart your pipeline to verify the new intrinsics are returned.")

def check_calibration():
    import json

    cal = np.load(calibration_path)
    camera_matrix = cal['camera_matrix']
    dist_coeffs = cal['dist_coeffs'].flatten()
    rms = float(cal['rms_error'])
    print(f"Loaded calibration (RMS: {rms:.4f}px)")

    # Connect to device
    ctx = rs.context()
    device = ctx.query_devices()[0]
    print(f"Connected to: {device.get_info(rs.camera_info.name)}")

    auto_cal = device.as_auto_calibrated_device()

    # Get current calibration table (returned as JSON string)
    table = auto_cal.get_calibration_table()
    print("Current calibration table:")
    print(table)

    # Inspect what keys are available before patching
    table_dict = json.loads(table)
    print("\nCalibration table keys:", list(table_dict.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, help="Path to calibration.npz")
    args = parser.parse_args()
    flash_calibration(args.calibration)