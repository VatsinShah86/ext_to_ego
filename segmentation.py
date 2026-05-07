import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import SAM


class Segmentation:
    """SAM2-based image segmentation producing per-pixel integer label maps.

    All inputs and outputs are numpy arrays — no dependency on vision.py.
    Images are expected as (H, W, 3) uint8 RGB.
    """

    def __init__(self, model_path: str = "sam2.1_b.pt"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SAM(model_path)
        self.model.to(self.device)

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run SAM2 on an RGB image and return a dense label map.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            label_map: (H, W) int32 array. 0 = background, 1..N = instance IDs.
                       If multiple masks overlap, the higher ID wins.
        """
        results = self.model(image_rgb, verbose=False)
        masks_obj = results[0].masks
        if masks_obj is None:
            return np.zeros(image_rgb.shape[:2], dtype=np.int32)

        masks = masks_obj.data.cpu().numpy().astype(bool)  # (N, H, W)
        label_map = np.zeros(image_rgb.shape[:2], dtype=np.int32)
        for i, mask in enumerate(masks):
            label_map[mask] = i + 1
        return label_map

    def lookup(self, label_map: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Look up segment IDs for a set of pixel coordinates.

        Args:
            label_map: (H, W) int32 from segment().
            uv:        (N, 2) array of (u, v) = (col, row) pixel coordinates.

        Returns:
            ids: (N,) int32 segment IDs, 0 where the point falls on background.
        """
        h, w = label_map.shape
        u = np.clip(uv[:, 0].astype(int), 0, w - 1)
        v = np.clip(uv[:, 1].astype(int), 0, h - 1)
        return label_map[v, u].astype(np.int32)

    def show(self, image_rgb: np.ndarray, label_map: np.ndarray) -> None:
        """Overlay the label map as coloured semi-transparent masks on the image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            label_map: (H, W) int32 from segment().
        """
        n = label_map.max()
        colors = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_rgb)
        for i in range(1, n + 1):
            overlay = np.zeros((*label_map.shape, 4))
            overlay[label_map == i] = [*colors[(i - 1) % len(colors)][:3], 0.45]
            ax.imshow(overlay)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from vision import RGBDData

    data = RGBDData("data/run_6_high_accuracy")

    rgb, _ = data.get_frame(0)

    seg = Segmentation()
    label_map = seg.segment(rgb)
    seg.show(rgb, label_map)