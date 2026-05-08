import pyrealsense2 as rs
import cv2
import numpy as np

# Reuse your DepthCamera class
from realsense_data_capture import DepthCamera

camera = DepthCamera(640,480)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y), SQUARE_SIZE, MARKER_SIZE, dictionary
)
detector = cv2.aruco.CharucoDetector(board)

all_charuco_corners = []
all_charuco_ids = []
image_size = None

print("Press SPACE to capture a frame, Q to finish")

while True:
    success, depth_frame, color_frame = camera.get_raw_frame()
    if not success:
        continue

    color_raw = np.asanyarray(color_frame.get_data())  # BGR
    gray = cv2.cvtColor(color_raw, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    # Draw detections on preview
    preview = color_raw.copy()
    if charuco_ids is not None and len(charuco_ids) >= 6:
        cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
        cv2.putText(preview, f"Detected {len(charuco_ids)} corners — SPACE to capture",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(preview, "Board not detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Calibration", preview)
    key = cv2.waitKey(1)

    if key == ord(' ') and charuco_ids is not None and len(charuco_ids) >= 6:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        print(f"Captured frame {len(all_charuco_corners)} ({len(charuco_ids)} corners)")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print(f"Calibration RMS error: {ret:.4f} px")  # good if < 1.0
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Save
np.savez("calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rms_error=ret
)

# RS2 intrinsics from your existing metadata
rs2_fx = metadata['color_intrinsics']['fx']
rs2_fy = metadata['color_intrinsics']['fy']
rs2_ppx = metadata['color_intrinsics']['ppx']
rs2_ppy = metadata['color_intrinsics']['ppy']

print("         fx      fy      cx      cy")
print(f"RS2:  {rs2_fx:.1f}  {rs2_fy:.1f}  {rs2_ppx:.1f}  {rs2_ppy:.1f}")
print(f"Cal:  {camera_matrix[0,0]:.1f}  {camera_matrix[1,1]:.1f}  "
      f"{camera_matrix[0,2]:.1f}  {camera_matrix[1,2]:.1f}")