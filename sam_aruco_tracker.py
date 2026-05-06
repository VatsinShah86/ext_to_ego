import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from vision import RGBDData


class ArucoSAMTracker:
    def __init__(self, checkpoint: str = "sam_vit_h_4b8939.pth", model_type: str = "vit_h"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

        # Set after extract_marker() is called
        self.template = None
        self.corners = None
        self.mask = None

    def extract_marker(self, image_rgb: np.ndarray, marker_x: int, marker_y: int,
                       box_size: int = 40) -> np.ndarray:
        """
        Given a manually specified point on the marker, segments it with SAM
        using a box prompt, extracts the 4 corners, and saves the marker crop
        as a template for future matching.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3)
            marker_x:  x pixel coordinate of the marker center
            marker_y:  y pixel coordinate of the marker center
            box_size:  side length of the box prompt around the click point.
                       Increase if the marker is large in the image.

        Returns:
            corners: (4, 2) array of corner pixel coordinates
        """
        self.predictor.set_image(image_rgb)

        x1 = marker_x - box_size // 2
        y1 = marker_y - box_size // 2
        x2 = marker_x + box_size // 2
        y2 = marker_y + box_size // 2

        masks, scores, _ = self.predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=True
        )

        self.mask = masks[np.argmax(scores)]
        self.corners = self._extract_corners(self.mask)

        # Save the tight crop as template for future template matching
        cx1, cy1 = self.corners.min(axis=0)
        cx2, cy2 = self.corners.max(axis=0)
        self.template = image_rgb[cy1:cy2, cx1:cx2]

        return self.corners

    def find_marker(self, image_rgb: np.ndarray, use_sam: bool = True,
                    rotation_search: bool = True) -> np.ndarray:
        """
        Uses template matching to locate the marker in a new frame,
        then optionally refines the segmentation with SAM using a box prompt.

        Args:
            image_rgb:        RGB image as numpy array (H, W, 3)
            use_sam:          if True, refines mask with SAM using the template
                              match bounding box as the prompt. If False, returns
                              the bounding box corners from template matching directly.
            rotation_search:  if True, searches ±45° in 5° increments to handle
                              marker rotation between frames.

        Returns:
            corners: (4, 2) array of corner pixel coordinates
        """
        if self.template is None:
            raise RuntimeError("No template saved. Call extract_marker() first.")

        best_val, best_loc, best_angle, best_template = self._template_search(
            image_rgb, rotation_search
        )

        if best_val < 0.6:
            print(f"Warning: low confidence match ({best_val:.2f}).")

        th, tw = best_template.shape[:2]
        x1, y1 = best_loc
        x2, y2 = x1 + tw, y1 + th
        print(f"Template match confidence: {best_val:.2f}, angle: {best_angle:.1f}°")
        print(f"Bounding box: ({x1}, {y1}) -> ({x2}, {y2})")

        if not use_sam:
            self.corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            return self.corners

        self.predictor.set_image(image_rgb)

        masks, scores, _ = self.predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=True
        )

        self.mask = masks[np.argmax(scores)]
        self.corners = self._extract_corners(self.mask)
        return self.corners

    def _template_search(self, image_rgb: np.ndarray,
                         rotation_search: bool) -> tuple:
        """
        Searches for the best template match, optionally across multiple rotations.

        Args:
            image_rgb:       RGB image to search in
            rotation_search: if True, searches ±45° in 5° increments

        Returns:
            best_val:      best match confidence score
            best_loc:      (x, y) top-left pixel of the best match
            best_angle:    rotation angle (degrees) that gave the best match
            best_template: the rotated template that gave the best match
        """
        best_val, best_loc, best_angle, best_template = 0, None, 0, self.template

        angles = np.arange(-45, 46, 5) if rotation_search else [0]

        for angle in angles:
            if angle == 0:
                rotated = self.template
            else:
                h, w = self.template.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                rotated = cv2.warpAffine(self.template, M, (w, h))

            result = cv2.matchTemplate(image_rgb, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_angle = angle
                best_template = rotated

        return best_val, best_loc, best_angle, best_template

    def _extract_corners(self, mask: np.ndarray) -> np.ndarray:
        """
        Converts a binary SAM mask into 4 corner points.

        Args:
            mask: boolean mask of shape (H, W)

        Returns:
            corners: (4, 2) array of corner pixel coordinates
        """
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) != 4:
            for eps_factor in np.linspace(0.01, 0.1, 20):
                approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
                if len(approx) == 4:
                    break

        if len(approx) != 4:
            print(f"approxPolyDP gave {len(approx)} corners, falling back to minAreaRect")
            rect = cv2.minAreaRect(contour)
            return np.int32(cv2.boxPoints(rect))

        return approx.reshape(4, 2)

    def show_marker(self, image_rgb: np.ndarray):
        """
        Displays the image with the extracted corners and mask overlaid.
        Requires extract_marker() or find_marker() to have been called first.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3)
        """
        if self.corners is None:
            raise RuntimeError("No marker found yet. Call extract_marker() first.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        debug = image_rgb.copy()
        for pt in self.corners:
            cv2.circle(debug, tuple(pt), 2, (0, 255, 0), -1)
        cv2.polylines(debug, [self.corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        axes[0].imshow(debug)
        axes[0].set_title("Detected Corners")
        axes[0].axis("off")

        axes[1].imshow(self.mask, cmap="gray")
        axes[1].set_title("SAM Mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = RGBDData("data/raw_camera_data_2")
    image_rgb, _ = data.get_frame(400)

    # tracker = ArucoSAMTracker()
    # corners = tracker.extract_marker(image_rgb, marker_x=365, marker_y=164)

    # frame_id_1 = 400
    # image_rgb_1, _ = data.get_frame(frame_id_1)
    # corners = tracker.find_marker(image_rgb_1)
    # print(f"Corners in frame {frame_id_1}:", corners)
    # tracker.show_marker(image_rgb_1)

    # image_rgb_5, _ = data.get_frame(5)
    # corners = tracker.find_marker(image_rgb_5)
    # print("Corners in frame 5:", corners)
    # tracker.show_marker(image_rgb_5)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model_type = "vit_h"
    checkpoint = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict()
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    for mask, color in zip(masks, colors):
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [*color, 0.4]
        ax.imshow(overlay)
    ax.axis("off")
    plt.show()