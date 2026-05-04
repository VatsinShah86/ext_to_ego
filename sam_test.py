import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from vision import RGBDData

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device)
predictor = SamPredictor(sam)

# Load image
data = RGBDData("data/raw_camera_data")
image_rgb, _ = data.get_frame(0)

# Inspect to find marker coordinates
plt.imshow(image_rgb)
plt.title("Note the marker center coordinates (x, y)")
plt.show()

# Set these after inspecting above
marker_x, marker_y = 232, 229  # <-- CHANGE THESE

# Run SAM
predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(
    point_coords=np.array([[marker_x, marker_y]]),
    point_labels=np.array([1]),
    multimask_output=True
)

mask = masks[np.argmax(scores)]
mask_uint8 = mask.astype(np.uint8) * 255

# Find 4 corners
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)
peri = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

if len(approx) != 4:
    for eps_factor in np.linspace(0.01, 0.1, 20):
        approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
        if len(approx) == 4:
            break

# Fallback: minAreaRect always gives exactly 4 corners
if len(approx) != 4:
    print(f"approxPolyDP gave {len(approx)} corners, falling back to minAreaRect")
    rect = cv2.minAreaRect(contour)
    approx = cv2.boxPoints(rect)  # always returns 4 corners
    corners = np.int32(approx)
else:
    corners = approx.reshape(4, 2)

print("4 corners:", corners)

# Visualize
debug = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
for pt in corners:
    cv2.circle(debug, tuple(pt), 5, (0, 255, 0), -1)
cv2.polylines(debug, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
cv2.imshow("Corners", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()