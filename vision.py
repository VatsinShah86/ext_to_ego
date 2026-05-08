import os
import cv2
import numpy as np
import zarr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from dataclasses import dataclass

class RGBDData:
    """Handles multi-frame RGBD data collected from a RealSense camera."""

    def __init__(self, folder: str):
        meta = np.load(os.path.join(folder, "metadata.npz"), allow_pickle=True)
        self.depth_scale: float = float(meta["depth_scale"])
        self.timestamps: np.ndarray = meta["timestamps"]
        self.recording_time: str = str(meta["recording_time"])
        
        # Load depth and color intrinsics from updated metadata
        self.depth_intrinsics: dict = meta["depth_intrinsics"].item()  # fx, fy, ppx, ppy, width, height, etc.
        self.color_intrinsics: dict = meta["color_intrinsics"].item()

        # For backward compatibility with code that uses self.intrinsics, use depth_intrinsics
        self.intrinsics: dict = self.depth_intrinsics

        intr = self.depth_intrinsics
        self.camera_matrix = np.array([
            [intr["fx"],  0.,          intr["ppx"]],
            [0.,          intr["fy"],  intr["ppy"]],
            [0.,          0.,          1.         ],
        ], dtype=np.float64)
        self.dist_coeffs = np.array(intr["coeffs"], dtype=np.float64)
        
        print(f"depth_scale: {self.depth_scale}")
        print(f"depth_intrinsics: {self.depth_intrinsics}")
        print(f"color_intrinsics: {self.color_intrinsics}")
        
        # Load from zarr (zarr doesn't support negative step slicing, so load to numpy first)
        color_zarr = zarr.open_array(os.path.join(folder, "color_raw.zarr"), mode='r')
        depth_zarr = zarr.open_array(os.path.join(folder, "depth_raw.zarr"), mode='r')
        
        # Load to numpy arrays and convert BGR to RGB
        self.color_frames: np.ndarray = np.asarray(color_zarr)[..., ::-1]  # Convert BGR to RGB
        self.depth_frames: np.ndarray = np.asarray(depth_zarr)

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

    def get_pointcloud(self, index: int, max_z: float = 10.0,
                       label_map: np.ndarray | None = None) -> np.ndarray:
        """Return an (N, 6) or (N, 7) float32 array for valid depth pixels.

        Columns: x, y, z  (metres, camera space)
                 r, g, b  (uint8 values as float32, 0–255)
                 seg_id   (int32 as float32, only if label_map is provided)

        Pixels with zero depth or z > max_z are excluded.
        """
        color_rgb, depth_m = self.get_frame(index)

        h, w = depth_m.shape
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))

        # Undistort pixel grid → normalized camera coordinates, then lift to 3D
        pts = np.stack([uu.ravel(), vv.ravel()], axis=-1).reshape(-1, 1, 2)
        pts_norm = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs).reshape(h, w, 2)

        z = depth_m
        x = pts_norm[..., 0] * z
        y = pts_norm[..., 1] * z

        valid = (z > 0) & (z <= max_z)
        xyz = np.stack([x, y, z], axis=-1)[valid]
        rgb = color_rgb[valid].astype(np.float32)

        if label_map is not None:
            seg_ids = label_map[valid].astype(np.float32).reshape(-1, 1)
            return np.concatenate([xyz, rgb, seg_ids], axis=-1)

        return np.concatenate([xyz, rgb], axis=-1)

    def plot_pointcloud(self, index: int, max_points: int = 30_000,
                        label_map: np.ndarray | None = None) -> None:
        """Plot the 3D point cloud for the given frame.

        If label_map is provided, each segment gets a randomly assigned colour
        for debugging — the actual point data is not modified.
        Uniformly subsamples to *max_points* when the cloud is larger.
        """
        pc = self.get_pointcloud(index, label_map=label_map)

        if len(pc) > max_points:
            idx = np.random.choice(len(pc), max_points, replace=False)
            pc = pc[idx]

        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        if label_map is not None:
            seg_ids = pc[:, 6].astype(int)
            unique_ids = np.unique(seg_ids)
            rng = np.random.default_rng(42)
            id_to_color = {uid: (rng.random(3) if uid > 0 else np.array([0.5, 0.5, 0.5]))
                           for uid in unique_ids}
            colors = np.array([id_to_color[sid] for sid in seg_ids])
            title_suffix = "  [segmentation colours]"
        else:
            colors = pc[:, 3:6] / 255.0
            title_suffix = ""

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=colors, s=0.5, linewidths=0)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Point cloud — frame {index}  |  t = {self.timestamps[index]:.3f} s"
                     f"  ({len(pc):,} pts shown){title_suffix}")
        plt.tight_layout()
        plt.show()

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


@dataclass
class MarkerDetection:
    corners: np.ndarray       # (4, 2) float32 — [TL, TR, BR, BL]
    center: np.ndarray        # (2,) float32 — pixel center
    angle_deg: float          # rotation of marker X-axis from image +X, degrees
    depth_samples: np.ndarray # (4,) float32 — depth at each corner in metres
    rvec: np.ndarray          # (3,) float64 — Rodrigues rotation vector (camera frame)
    tvec: np.ndarray          # (3,) float64 — translation vector in metres (camera frame)


class ArucoTracker:
    """Detects and plots an ArUco marker (id=0) in RGBDData frames.

    The physical marker uses the DICT_4X4_50 encoding.
    """

    _DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def __init__(self, rgbd: RGBDData, marker_size_m: float = 0.0468):
        self.rgbd = rgbd
        self.marker_size_m = marker_size_m

        intr = rgbd.depth_intrinsics
        self.camera_matrix = np.array([
            [intr["fx"],  0.,          intr["ppx"]],
            [0.,          intr["fy"],  intr["ppy"]],
            [0.,          0.,          1.         ],
        ], dtype=np.float64)
        self.dist_coeffs = rgbd.dist_coeffs.reshape(-1, 1)

        # 3D corners in marker space [TL, TR, BR, BL], Y-up as required by IPPE_SQUARE
        h = marker_size_m / 2
        self._obj_pts = np.array([
            [-h,  h, 0.],
            [ h,  h, 0.],
            [ h, -h, 0.],
            [-h, -h, 0.],
        ], dtype=np.float64)

        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.01
        params.errorCorrectionRate = 1.0
        self._detector = cv2.aruco.ArucoDetector(self._DICT, params)

        # URDF joint (camera_link → aruco_link): rpy=[-π/2, 0, -π/2], xyz=[0, 0, -0.02]
        # R_C_A = Rz(-π/2) @ Ry(0) @ Rx(-π/2)
        r, y = -np.pi / 2, -np.pi / 2
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        R_C_A = Rz @ Rx
        t_C_A = np.array([0.0, 0.0, -0.02])
        # Invert → aruco frame → camera frame
        self._R_aruco_camera: np.ndarray = R_C_A.T
        self._t_aruco_camera: np.ndarray = -R_C_A.T @ t_C_A

    def camera_in_aruco_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R, t): orientation and position of the camera origin in the aruco frame.

        Derived by inverting the URDF joint (camera_link → aruco_link,
        rpy=[-π/2, 0, -π/2], xyz=[0, 0, -0.02]).
        """
        return self._R_aruco_camera, self._t_aruco_camera

    def get_camera_pose(self, det: MarkerDetection) -> tuple[np.ndarray, np.ndarray]:
        """Return (R, t): camera frame orientation and origin expressed in camera space.

        Uses the detected aruco pose together with the URDF aruco→camera transform
        to express where the camera coordinate frame sits in 3D camera space.
        """
        R_aruco, _ = cv2.Rodrigues(det.rvec)
        R_A_C, t_A_C = self.camera_in_aruco_frame()
        R = R_aruco @ R_A_C
        t = R_aruco @ t_A_C + det.tvec
        return R, t

    def detect(self, index: int) -> MarkerDetection | None:
        """Return MarkerDetection for marker id=0 in frame *index*, or None if not found."""
        bgr = self.rgbd.color_frames[index][..., ::-1]
        corners, ids, _ = self._detector.detectMarkers(bgr)

        if ids is None:
            return None

        ids_flat = ids.flatten()
        matches = [i for i, mid in enumerate(ids_flat) if mid == 0]
        if not matches:
            return None

        marker_corners = corners[matches[0]][0]  # (4, 2) float32 — [TL, TR, BR, BL]
        center = marker_corners.mean(axis=0)

        # Angle of marker X-axis (TL→TR) relative to image +X axis
        dx, dy = marker_corners[1] - marker_corners[0]
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        depth_frame = self.rgbd.depth_frames[index]
        pts = np.round(marker_corners).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, depth_frame.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, depth_frame.shape[0] - 1)
        depth_samples = depth_frame[pts[:, 1], pts[:, 0]].astype(np.float32) * self.rgbd.depth_scale

        _, rvec, tvec = cv2.solvePnP(
            self._obj_pts,
            marker_corners.astype(np.float64),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        rvec, tvec = rvec.flatten(), tvec.flatten()

        # solvePnP can drift along the marker's own Z-axis (normal).
        # Correct it by snapping to the depth-measured surface:
        # slide tvec along the marker normal until it agrees with the back-projected depth centre.
        valid = depth_samples[depth_samples > 0]
        if len(valid):
            z = float(valid.mean())
            center_pt = np.array([[[float(center[0]), float(center[1])]]], dtype=np.float64)
            center_norm = cv2.undistortPoints(center_pt, self.camera_matrix, self.dist_coeffs).flatten()
            p_depth = np.array([center_norm[0] * z, center_norm[1] * z, z])
            R, _ = cv2.Rodrigues(rvec)
            marker_normal = R[:, 2]
            tvec = tvec + float(marker_normal @ (p_depth - tvec)) * marker_normal

        return MarkerDetection(
            corners=marker_corners,
            center=center,
            angle_deg=angle_deg,
            depth_samples=depth_samples,
            rvec=rvec,
            tvec=tvec,
        )

    def detect_plane(self, index: int) -> MarkerDetection | None:
        """Detect the marker pose by fitting a plane to back-projected boundary pixels.

        Uses detect() for the 2D corners, then rasterises the 4 marker edges,
        back-projects every boundary pixel to 3D via depth, and fits a plane by SVD.
        The fitted plane's normal becomes the marker Z-axis; the in-plane X-axis is
        derived from the back-projected TL→TR corner vector projected onto the plane.
        Falls back to detect() if there are too few valid depth pixels on the boundary.
        """
        det = self.detect(index)
        if det is None:
            return None

        depth_frame = self.rgbd.depth_frames[index]
        h_img, w_img = depth_frame.shape

        # Rasterise the 4 boundary edges
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.polylines(mask, [det.corners.astype(np.int32).reshape(-1, 1, 2)],
                      isClosed=True, color=255, thickness=2)
        ys, xs = np.where(mask > 0)

        # Back-project boundary pixels to 3D
        z = depth_frame[ys, xs].astype(np.float64) * self.rgbd.depth_scale
        valid = z > 0
        if valid.sum() < 6:
            return det  # not enough depth readings — fall back

        xs, ys, z = xs[valid].astype(np.float64), ys[valid].astype(np.float64), z[valid]
        pts_2d = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
        pts_norm = cv2.undistortPoints(pts_2d, self.camera_matrix, self.dist_coeffs).reshape(-1, 2)
        pts3d = np.stack([pts_norm[:, 0] * z, pts_norm[:, 1] * z, z], axis=1)

        # Fit plane via SVD — last right-singular vector is the plane normal
        centroid = pts3d.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts3d - centroid, full_matrices=False)
        normal = Vt[-1]
        if normal[2] > 0:   # ensure normal points toward camera (−Z in camera frame)
            normal = -normal

        # Build X-axis from back-projected TL→TR, projected onto the fitted plane
        def backproject(corner_idx: int) -> np.ndarray | None:
            cx, cy = det.corners[corner_idx]
            ci, ri = int(np.clip(cx, 0, w_img - 1)), int(np.clip(cy, 0, h_img - 1))
            d = depth_frame[ri, ci].astype(float) * self.rgbd.depth_scale
            if d <= 0:
                return None
            pt = np.array([[[float(cx), float(cy)]]], dtype=np.float64)
            norm = cv2.undistortPoints(pt, self.camera_matrix, self.dist_coeffs).flatten()
            return np.array([norm[0] * d, norm[1] * d, d])

        p_TL, p_TR = backproject(0), backproject(1)
        if p_TL is not None and p_TR is not None:
            x_cand = p_TR - p_TL
        else:
            angle_rad = np.radians(det.angle_deg)
            x_cand = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])

        x_axis = x_cand - np.dot(x_cand, normal) * normal
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)   # enforce right-handed: Z = X × Y
        z_axis /= np.linalg.norm(z_axis)

        R_plane = np.column_stack([x_axis, y_axis, z_axis])
        rvec_plane, _ = cv2.Rodrigues(R_plane)

        return MarkerDetection(
            corners=det.corners,
            center=det.center,
            angle_deg=det.angle_deg,
            depth_samples=det.depth_samples,
            rvec=rvec_plane.flatten(),
            tvec=centroid,
        )

    def debug_frame(self, det: MarkerDetection) -> None:
        """Print orthogonality and handedness diagnostics for a MarkerDetection frame."""
        R, _ = cv2.Rodrigues(det.rvec)
        x, y, z = R[:, 0], R[:, 1], R[:, 2]
        print(f"  |x|={np.linalg.norm(x):.6f}  |y|={np.linalg.norm(y):.6f}  |z|={np.linalg.norm(z):.6f}  (all should be 1)")
        print(f"  x·y={np.dot(x,y):.6f}  x·z={np.dot(x,z):.6f}  y·z={np.dot(y,z):.6f}  (all should be 0)")
        cross_xy = np.cross(x, y)
        print(f"  det(R)={np.linalg.det(R):.6f}  (should be +1)")
        print(f"  x×y≈z: {np.allclose(cross_xy, z, atol=1e-4)}  max_err={np.max(np.abs(cross_xy - z)):.2e}")

    def plot_frame(self, index: int) -> None:
        """Plot color image and depth map for *index* with the detected marker overlaid."""
        det = self.detect(index)
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

        if det is not None:
            closed = np.vstack([det.corners, det.corners[0]])
            cx, cy = det.center

            # Arrow length = half the marker's top-edge length
            arrow_len = float(np.linalg.norm(det.corners[1] - det.corners[0])) * 0.5
            angle_rad = np.radians(det.angle_deg)
            # Marker X-axis (red) and Y-axis (green) in image coordinates
            x_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * arrow_len
            y_vec = np.array([-np.sin(angle_rad), np.cos(angle_rad)]) * arrow_len

            mean_depth = det.depth_samples[det.depth_samples > 0].mean() \
                if (det.depth_samples > 0).any() else float("nan")

            for ax in axes:
                ax.plot(closed[:, 0], closed[:, 1], "w-", linewidth=1.5, alpha=0.8)
                ax.scatter(det.corners[:, 0], det.corners[:, 1], c="white", s=25, zorder=5)
                ax.annotate("", xy=(cx + x_vec[0], cy + x_vec[1]), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
                ax.annotate("", xy=(cx + y_vec[0], cy + y_vec[1]), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="-|>", color="lime", lw=2))

            axes[0].annotate(
                f"id=0  {det.angle_deg:.1f}°  {mean_depth:.3f} m",
                xy=(cx, cy - arrow_len - 6), color="white", fontsize=9,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
            )
            legend = [
                mpatches.Patch(color="red",  label="marker X"),
                mpatches.Patch(color="lime", label="marker Y"),
            ]
            axes[0].legend(handles=legend, loc="upper right", fontsize=8,
                           framealpha=0.6, facecolor="black", labelcolor="white")
        else:
            for ax in axes:
                ax.set_title(ax.get_title() + "  [marker not found]")

        plt.tight_layout()
        plt.show()

    def plot_pointcloud_with_marker(self, index: int, radius: float = 0.20) -> None:
        """Plot points within *radius* metres of the marker centre with two frames overlaid.

        Red/green/blue    — aruco frame  (detect_plane)
        Orange/purple/brown — camera frame (URDF aruco→camera transform)
        """
        det_plane = self.detect_plane(index)
        pc = self.rgbd.get_pointcloud(index)

        center = det_plane.tvec if det_plane is not None else None
        if center is not None:
            dist = np.linalg.norm(pc[:, :3] - center, axis=1)
            pc = pc[dist <= radius]

        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                   c=pc[:, 3:] / 255.0, s=1, linewidths=0)

        axis_len = self.marker_size_m * 1.5

        def _draw_frame(origin: np.ndarray, R: np.ndarray,
                        axis_colors: tuple, label_suffix: str,
                        square: bool = False) -> None:
            for vec, color, label in zip(R.T, axis_colors, ("X", "Y", "Z")):
                end = origin + vec * axis_len
                ax.quiver(*origin, *(end - origin),
                          color=color, linewidth=2, arrow_length_ratio=0.2)
                ax.text(*end, f"{label}{label_suffix}", color=color, fontsize=8)
            if square:
                cam_corners = (R @ self._obj_pts.T).T + origin
                sq = np.vstack([cam_corners, cam_corners[0]])
                ax.plot(sq[:, 0], sq[:, 1], sq[:, 2], "w-", linewidth=1.5, alpha=0.9)

        if det_plane is not None:
            R_aruco, _ = cv2.Rodrigues(det_plane.rvec)
            # Aruco frame
            _draw_frame(det_plane.tvec, R_aruco,
                        axis_colors=("red", "green", "blue"),
                        label_suffix="", square=True)

            # Camera frame expressed in camera space via URDF aruco→camera transform
            R_cam_in_cam, cam_origin = self.get_camera_pose(det_plane)
            _draw_frame(cam_origin, R_cam_in_cam,
                        axis_colors=("orange", "purple", "saddlebrown"),
                        label_suffix="c")

            origin = det_plane.tvec
            ax.set_title(
                f"Frame {index}  |  t = {self.rgbd.timestamps[index]:.3f} s\n"
                f"aruco @ ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}) m  "
                f"  red/green/blue = aruco    orange/purple/brown = camera (URDF)"
            )
        else:
            ax.set_title(f"Frame {index}  [marker not found]")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        if len(pc):
            mid = pc[:, :3].mean(axis=0)
            half = (pc[:, :3].max(axis=0) - pc[:, :3].min(axis=0)).max() / 2
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    data = RGBDData("data/run_6_high_accuracy")

    # for i in range (10):
    #     data.plot_pointcloud(3*i)

    data.plot_frame(30)
    # data.plot_pointcloud(0, 300000)
    # aruco_tracker = ArucoTracker(data)
    # aruco_tracker.plot_frame(500)
    # data.plot_pointcloud(0, 100000)
    # start_ns = time.perf_counter_ns()
    # aruco_tracker.detect(500)
    # elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    # print(f"detect(500) took {elapsed_ms:.3f} ms")
    
    # aruco_tracker.plot_pointcloud_with_marker(0, 0.5)
