import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class RGBDData:
    """Handles multi-frame RGBD data collected from a RealSense camera."""

    def __init__(self, folder: str):
        meta = np.load(os.path.join(folder, "metadata.npz"), allow_pickle=True)

        self.depth_scale: float = float(meta["depth_scale"])
        self.timestamps: np.ndarray = meta["timestamps"]
        self.recording_time: str = str(meta["recording_time"])
        self.intrinsics: dict = meta["intrinsics"].item()  # fx, fy, ppx, ppy, width, height

        self.color_frames: np.ndarray = np.load(os.path.join(folder, "color_raw.npy"))[..., ::-1]
        self.depth_frames: np.ndarray = np.load(os.path.join(folder, "depth_raw.npy"))

        if self.color_frames.shape[0] != self.depth_frames.shape[0]:
            raise ValueError(
                f"Frame count mismatch: color has {self.color_frames.shape[0]}, "
                f"depth has {self.depth_frames.shape[0]}"
            )

    @property
    def num_frames(self) -> int:
        return self.color_frames.shape[0]

    def get_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (color_rgb, depth_m) for the given frame index."""
        if not (0 <= index < self.num_frames):
            raise IndexError(f"Frame index {index} out of range [0, {self.num_frames - 1}]")
        return self.color_frames[index], self.depth_frames[index].astype(np.float32) * self.depth_scale

    def plot_frame(self, index: int) -> None:
        """Plot the color image and depth map (in metres) for a given frame index."""
        if not (0 <= index < self.num_frames):
            raise IndexError(f"Frame index {index} out of range [0, {self.num_frames - 1}]")

        color_rgb, depth_m = self.get_frame(index)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Frame {index}  |  t = {self.timestamps[index]:.3f} s")

        axes[0].imshow(color_rgb)
        axes[0].set_title("Color (RGB)")
        axes[0].axis("off")

        im = axes[1].imshow(depth_m, cmap="plasma")
        axes[1].set_title("Depth (m)")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


class ArucoTracker:
    """Detects and plots an ArUco marker (id=0) in RGBDData frames.

    The physical marker uses the DICT_ARUCO_MIP_36h12 encoding.
    """

    _DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)

    def __init__(self, rgbd: RGBDData):
        self.rgbd = rgbd

        intr = rgbd.intrinsics
        self.camera_matrix = np.array([
            [intr["fx"],  0.,          intr["ppx"]],
            [0.,          intr["fy"],  intr["ppy"]],
            [0.,          0.,          1.         ],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.01
        params.errorCorrectionRate = 1.0
        self._detector = cv2.aruco.ArucoDetector(self._DICT, params)

    def detect(self, index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (corners, depth_samples) for marker id=0 in frame *index*, or (None, None)."""
        bgr = self.rgbd.color_frames[index][..., ::-1]
        corners, ids, _ = self._detector.detectMarkers(bgr)

        if ids is None:
            return None, None

        ids_flat = ids.flatten()
        matches = [i for i, mid in enumerate(ids_flat) if mid == 0]
        if not matches:
            return None, None

        marker_corners = corners[matches[0]][0]  # (4, 2) float32

        depth_frame = self.rgbd.depth_frames[index]
        pts = np.round(marker_corners).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, depth_frame.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, depth_frame.shape[0] - 1)
        depth_samples = depth_frame[pts[:, 1], pts[:, 0]].astype(np.float32) * self.rgbd.depth_scale

        return marker_corners, depth_samples

    def plot_frame(self, index: int) -> None:
        """Plot color image and depth map for *index* with the detected marker overlaid."""
        corners, depth_samples = self.detect(index)

        color_rgb, depth_m = self.rgbd.get_frame(index)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Frame {index}  |  t = {self.rgbd.timestamps[index]:.3f} s")

        axes[0].imshow(color_rgb)
        axes[0].set_title("Color (RGB)")
        axes[0].axis("off")

        im = axes[1].imshow(depth_m, cmap="plasma")
        axes[1].set_title("Depth (m)")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        if corners is not None:
            closed = np.vstack([corners, corners[0]])
            for ax in axes:
                ax.plot(closed[:, 0], closed[:, 1], "g-", linewidth=2)
                ax.scatter(corners[:, 0], corners[:, 1], c="lime", s=40, zorder=5)

            cx, cy = corners.mean(axis=0)
            mean_depth = depth_samples[depth_samples > 0].mean() if (depth_samples > 0).any() else float("nan")
            axes[0].annotate(
                f"id=0  {mean_depth:.3f} m",
                xy=(cx, cy), color="white", fontsize=9,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
            )
        else:
            for ax in axes:
                ax.set_title(ax.get_title() + "  [marker not found]")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = RGBDData("data/raw_camera_data")

    for i in range (10):
        data.plot_frame(100*i)
    # aruco_tracker = ArucoTracker(data)
    # aruco_tracker.plot_frame(0)