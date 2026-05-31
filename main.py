import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import zarr
from scipy.spatial import cKDTree
from vision import RGBDData, ArucoTracker, MarkerDetection
from realtime_segmentation import RealtimeSegmentation


def create_example_pc(density: float = 10):
    """
    Create a flat example point cloud on the ground plane.

    Args:
        density: Sampling density in points per millimeter.

    Returns:
        points: (N, 3) float array of ground-plane points in world frame.
    """
    if density <= 0:
        raise ValueError("density must be positive")

    ground_size_mm = 1
    step_mm = 1.0 / density
    half_size_mm = ground_size_mm * 0.5

    coords = np.arange(
        -half_size_mm,
        half_size_mm + (step_mm * 0.5),
        step_mm,
        dtype=np.float32,
    )
    point_count = coords.size

    total_points = point_count * point_count
    points = np.empty((total_points, 3), dtype=np.float32)
    points[:, 0] = np.repeat(coords, point_count)
    points[:, 1] = np.tile(coords, point_count)
    points[:, 2] = 0.0
    return points


def create_box_pc(center: np.ndarray, length: float, width: float, height: float, density: float = 10) -> np.ndarray:
    """
    Create a hollow box point cloud sampled on its outer surface.

    Args:
        center:  (3,) box center in world coordinates.
        length:  Box size along the x-axis.
        width:   Box size along the y-axis.
        height:  Box size along the z-axis.
        density: Sampling density in points per millimeter.

    Returns:
        points: (N, 3) float array of points on the box surface.
    """
    center = np.asarray(center, dtype=np.float32)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    if density <= 0:
        raise ValueError("density must be positive")
    if length <= 0 or width <= 0 or height <= 0:
        raise ValueError("length, width, and height must be positive")

    step = 1.0 / density

    def make_axis_coords(size: float) -> np.ndarray:
        half_size = size * 0.5
        coords = np.arange(
            -half_size,
            half_size + (step * 0.5),
            step,
            dtype=np.float32,
        )
        if coords.size == 0:
            coords = np.array([-half_size], dtype=np.float32)
        if not np.isclose(coords[-1], half_size):
            coords = np.append(coords, np.float32(half_size))
        return coords

    x_coords = make_axis_coords(length) + center[0]
    y_coords = make_axis_coords(width) + center[1]
    z_coords = make_axis_coords(height) + center[2]

    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]
    z_min, z_max = z_coords[0], z_coords[-1]

    yz_y, yz_z = np.meshgrid(y_coords, z_coords, indexing="ij")
    xz_x, xz_z = np.meshgrid(x_coords, z_coords, indexing="ij")
    xy_x, xy_y = np.meshgrid(x_coords, y_coords, indexing="ij")

    faces = [
        np.column_stack([np.full(yz_y.size, x_min, dtype=np.float32), yz_y.ravel(), yz_z.ravel()]),
        np.column_stack([np.full(yz_y.size, x_max, dtype=np.float32), yz_y.ravel(), yz_z.ravel()]),
        np.column_stack([xz_x.ravel(), np.full(xz_x.size, y_min, dtype=np.float32), xz_z.ravel()]),
        np.column_stack([xz_x.ravel(), np.full(xz_x.size, y_max, dtype=np.float32), xz_z.ravel()]),
        np.column_stack([xy_x.ravel(), xy_y.ravel(), np.full(xy_x.size, z_min, dtype=np.float32)]),
        np.column_stack([xy_x.ravel(), xy_y.ravel(), np.full(xy_x.size, z_max, dtype=np.float32)]),
    ]

    points = np.vstack(faces).astype(np.float32, copy=False)
    return np.unique(points, axis=0)


def plot_pc(pc: np.ndarray):
    """
    Display a 3D scatter plot for a point cloud.

    Args:
        pc: (N, 3) xyz or (N, 6) xyzrgb array. RGB values are expected in
            the 0–255 range and are normalised to 0–1 for display.
    """
    pc = np.asarray(pc)
    assert pc.ndim == 2 and pc.shape[1] in (3, 6, 7), "pc must have shape (N, 3), (N, 6), or (N, 7)"

    xyz    = pc[:, :3]
    colors = pc[:, 3:6] / 255.0 if pc.shape[1] >= 6 else None

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=1, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if pc.shape[0] > 0:
        pc = xyz
        mins = pc.min(axis=0)
        maxs = pc.max(axis=0)

        xy_span = max(maxs[0] - mins[0], maxs[1] - mins[1])
        if xy_span == 0:
            xy_span = 1.0

        half_xy_span = 0.5 * xy_span
        x_center = 0.5 * (mins[0] + maxs[0])
        y_center = 0.5 * (mins[1] + maxs[1])
        z_min = min(mins[2], 0.0)
        z_max = max(maxs[2], 1.0)
        z_span = z_max - z_min

        ax.set_xlim(x_center - half_xy_span, x_center + half_xy_span)
        ax.set_ylim(y_center - half_xy_span, y_center + half_xy_span)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect((xy_span, xy_span, z_span))

    plt.show()


class Ext2Ego:
    """Full pipeline from an external RGBD camera to an egocentric camera frame.

    Chains RGBDData point cloud generation, ArucoTracker pose estimation,
    coordinate transform, frustum culling, and occlusion culling into a
    single per-frame process() call.
    """

    def __init__(self, rgbd: RGBDData | None, tracker: ArucoTracker, config_path: str,
                 segmentation: RealtimeSegmentation | None = None):
        self.rgbd         = rgbd
        self.tracker      = tracker
        self.segmentation = segmentation
        self.T_cw: np.ndarray | None = None

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["camera"]

        W  = cfg["resolution"]["width"]
        H  = cfg["resolution"]["height"]
        fx = cfg["intrinsics"]["focal_length"]["fx"]
        fy = cfg["intrinsics"]["focal_length"]["fy"]
        cx = cfg["intrinsics"]["principal_point"]["cx"]
        cy = cfg["intrinsics"]["principal_point"]["cy"]

        self.cam_name = cfg["name"]
        self.width    = W
        self.height   = H
        self.fx       = fx
        self.fy       = fy
        self.cx       = cx
        self.cy       = cy
        self.z_near   = cfg["clip"]["near"]
        self.z_far    = cfg["clip"]["far"]

        self._tan_h = np.tan(np.radians(np.degrees(2 * np.arctan(W / (2 * fx))) / 2))
        self._tan_v = np.tan(np.radians(np.degrees(2 * np.arctan(H / (2 * fy))) / 2))

    def set_pose(self, R_wc: np.ndarray, camera_pos: np.ndarray) -> None:
        """Set the ego camera pose from get_camera_pose() output.

        R_wc has columns = ego camera axes in the RealSense frame.
        camera_pos is the ego camera origin in the RealSense frame.
        """
        R = R_wc.T
        t = -R @ camera_pos
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = t
        self.T_cw = T

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Transform (N, 3) points from RealSense frame to ego camera frame."""
        if self.T_cw is None:
            raise RuntimeError("Call set_pose() before transform().")
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        return (self.T_cw @ pts_h.T).T[:, :3]

    def filter_frustum(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (visible_points, mask) after frustum culling."""
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        mask = (
            (z >  0)                &
            (z >= self.z_near)      &
            (z <= self.z_far)       &
            (x <=  z * self._tan_h) &
            (x >= -z * self._tan_h) &
            (y <=  z * self._tan_v) &
            (y >= -z * self._tan_v)
        )
        return points_cam[mask], mask

    def cull_occlusion(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (visible_points, mask) using depth-buffer rasterisation."""
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

        u = (self.fx * x / z + self.cx).astype(int)
        v = (self.fy * y / z + self.cy).astype(int)

        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u, v, z   = u[in_bounds], v[in_bounds], z[in_bounds]
        orig_idx  = np.where(in_bounds)[0]

        depth_buf = np.full((self.height, self.width), np.inf)
        np.minimum.at(depth_buf, (v, u), z)

        visible_local = np.isclose(z, depth_buf[v, u])
        mask = np.zeros(len(points_cam), dtype=bool)
        mask[orig_idx[visible_local]] = True
        return points_cam[mask], mask

    def cull_occlusion_soft(
        self, points_cam: np.ndarray, pixel_radius: float = 12.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (visible_points, mask) using KD-tree soft occlusion culling."""
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u, v, z_in = u[in_bounds], v[in_bounds], z[in_bounds]
        orig_idx   = np.where(in_bounds)[0]

        tree = cKDTree(np.stack([u, v], axis=1))

        mask_local = np.ones(len(u), dtype=bool)
        for i, (ui, vi, zi) in enumerate(zip(u, v, z_in)):
            neighbor_z = z_in[tree.query_ball_point([ui, vi], r=pixel_radius)]
            if np.any(neighbor_z < zi - 1e-3):
                mask_local[i] = False

        mask = np.zeros(len(points_cam), dtype=bool)
        mask[orig_idx[mask_local]] = True
        return points_cam[mask], mask

    def process(
        self, index: int, occlusion: bool = False
    ) -> tuple[np.ndarray, MarkerDetection | None]:
        """Run the full pipeline for frame *index*.

        Args:
            index:     Frame index into the RGBD dataset.
            occlusion: If False, skip occlusion culling (faster, useful for testing).

        Returns (visible_points, det) where visible_points is an (M, 6) xyzrgb
        or (M, 7) xyzrgb+seg_id array in ego camera frame, and det is the
        MarkerDetection used to derive the ego camera pose (None if not found).
        """
        color_rgb, depth_m = self.rgbd.get_frame(index)
        label_map = None
        if self.segmentation is not None:
            label_map, _ = self.segmentation.process_frame(color_rgb)

        pc  = RGBDData.get_pointcloud_from_arrays(
            color_rgb, depth_m,
            self.tracker.camera_matrix, self.tracker.dist_coeffs,
            label_map=label_map,
        )
        det = self.tracker.detect_plane(index)
        self.rgbd.plot_pointcloud(index, label_map=label_map)

        n_extra = pc.shape[1] - 6  # 0 without seg, 1 with
        if det is None:
            return np.empty((0, 6 + n_extra), dtype=np.float32), None

        R, t = self.tracker.get_camera_pose(det)
        self.set_pose(R, t)

        xyz_cam         = self.transform(pc[:, :3])
        _, frustum_mask = self.filter_frustum(xyz_cam)

        xyz_frustum   = xyz_cam[frustum_mask]
        extra_frustum = pc[frustum_mask, 3:]          # rgb (+ seg_id if present)
        result        = np.concatenate([xyz_frustum, extra_frustum], axis=1)

        if occlusion:
            _, occ_mask = self.cull_occlusion(xyz_frustum)
            result      = result[occ_mask]

        return result, det

    def detect_from_sam(
        self, index: int, prev_det: MarkerDetection
    ) -> MarkerDetection | None:
        """Recover a MarkerDetection via SAM when ArUco detection fails.

        Uses the bounding box of prev_det.corners as a SAM bbox prompt to
        isolate the marker region, then fits a plane to the segmented depth
        pixels via detect_plane_from_mask().

        Args:
            index:    Current frame index.
            prev_det: Last successful MarkerDetection — corners define the SAM
                      bbox prompt; angle_deg seeds the fitted plane's X-axis.

        Returns:
            MarkerDetection from the SAM plane fit, or None if segmentation
            returns an empty mask or there are too few valid depth pixels.
        """
        if self.segmentation is None:
            return None
        color_rgb, _ = self.rgbd.get_frame(index)
        corners = prev_det.corners                          # (4, 2) float32
        pad = 5.0
        bbox = np.array([
            corners[:, 0].min() - pad, corners[:, 1].min() - pad,
            corners[:, 0].max() + pad, corners[:, 1].max() + pad,
        ])
        mask = self.segmentation.segment_box(color_rgb, bbox)
        if not mask.any():
            return None
        return self.tracker.detect_plane_from_mask(index, mask, prev_det)

    def detect_from_sam_frame(
        self, color_rgb: np.ndarray, depth_m: np.ndarray, prev_det: MarkerDetection
    ) -> MarkerDetection | None:
        """SAM fallback for a live frame — mirrors detect_from_sam() but takes arrays."""
        if self.segmentation is None:
            return None
        corners = prev_det.corners
        pad  = 5.0
        bbox = np.array([
            corners[:, 0].min() - pad, corners[:, 1].min() - pad,
            corners[:, 0].max() + pad, corners[:, 1].max() + pad,
        ])
        mask = self.segmentation.segment_box(color_rgb, bbox)
        if not mask.any():
            return None
        return self.tracker.detect_plane_from_mask_frame(depth_m, mask, prev_det)

    def process_live(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        prev_det: MarkerDetection | None = None,
        label_map: np.ndarray | None = None,
        occlusion: bool = False,
        num_pts = 1024,
        label_percentages: dict[int, float] | None = None,
    ) -> tuple[np.ndarray, MarkerDetection | None]:
        """Process a single live RGBD frame into a (num_pts, 7) ego-frame point cloud.

        Args:
            color_rgb:          (H, W, 3) uint8 RGB — convert from RealSense BGR before calling.
            depth_m:            (H, W) float32 depth in metres, aligned to color.
            prev_det:           Last good MarkerDetection; enables SAM fallback when ArUco is occluded.
            label_map:          (H, W) int32 segmentation labels, or None to run per-frame SAM2.
            occlusion:          Whether to apply occlusion culling (slower).
            num_pts:            Number of points to sample from the visible cloud.
            label_percentages:  When provided, frustum culling is skipped entirely.
                                Pass None to keep the default frustum-culled behaviour.

        Returns:
            (pc, det) — pc is (num_pts, 7) float32 in ego frame; det is the MarkerDetection
            used this frame (pass back as prev_det on the next call).
            Returns (zeros, None) if marker detection fails completely.
        """
        if label_map is None and self.segmentation is not None:
            label_map, _ = self.segmentation.process_frame(color_rgb)

        pc  = RGBDData.get_pointcloud_from_arrays(
            color_rgb, depth_m,
            self.tracker.camera_matrix,
            self.tracker.dist_coeffs,
            label_map=label_map,
            num_pts=num_pts if label_percentages is not None else None,
            label_percentages=label_percentages,
        )

        det = self.tracker.detect_plane_from_frame(color_rgb, depth_m)

        if det is not None:
            prev_det = det
        elif prev_det is not None:
            det = self.detect_from_sam_frame(color_rgb, depth_m, prev_det)
            if det is not None:
                prev_det = det

        if det is None:
            return np.zeros((num_pts, 7), dtype=np.float32), None

        R, t = self.tracker.get_camera_pose(det)
        self.set_pose(R, t)

        xyz_cam = self.transform(pc[:, :3])

        if label_percentages is not None:
            # Stratified sampling already done in get_pointcloud_from_arrays; no frustum cull.
            return np.concatenate([xyz_cam, pc[:, 3:]], axis=1), det

        _, frustum_mask = self.filter_frustum(xyz_cam)
        xyz_frustum     = xyz_cam[frustum_mask]
        extra_frustum   = pc[frustum_mask, 3:]
        result          = np.concatenate([xyz_frustum, extra_frustum], axis=1)

        if occlusion:
            _, occ_mask = self.cull_occlusion(xyz_frustum)
            result      = result[occ_mask]

        if result.shape[1] == 6:
            result = np.concatenate(
                [result, np.zeros((len(result), 1), dtype=np.float32)], axis=1
            )

        if len(result) == 0:
            return np.zeros((num_pts, 7), dtype=np.float32), det

        idx = np.round(np.linspace(0, len(result) - 1, num_pts)).astype(int)
        return result[idx], det

    def process_episode(
        self,
        num_pts = 1024,
        label_percentages: dict[int, float] | None = None,
    ) -> np.ndarray | None:
        """Build an ego-frame point cloud for every frame in the episode.

        Args:
            num_pts:            Number of points to sample per frame.
            label_percentages:  When provided, frustum culling is skipped entirely.
                                Pass None to keep the default frustum-culled behaviour.
        """
        N         = self.rgbd.num_frames
        pc_arrays = np.zeros((N, num_pts, 7), dtype=np.float32)

        prev_det = None
        for index in range(N):
            color_rgb, depth_m = self.rgbd.get_frame(index)
            label_map = None
            if self.segmentation is not None:
                label_map, _ = self.segmentation.process_frame(color_rgb)

            pc  = RGBDData.get_pointcloud_from_arrays(
                color_rgb, depth_m,
                self.tracker.camera_matrix, self.tracker.dist_coeffs,
                label_map=label_map,
                num_pts=num_pts if label_percentages is not None else None,
                label_percentages=label_percentages,
            )
            det = self.tracker.detect_plane(index)

            if det is not None:
                prev_det = det
            elif prev_det is not None:
                det = self.detect_from_sam(index, prev_det)
                if det is not None:
                    prev_det = det

            if det is None:
                # Catastrophic failure, ArUco not visible at all.
                print(f"Aruco detection failed at index {index}")
                return None

            R, t = self.tracker.get_camera_pose(det)
            self.set_pose(R, t)

            xyz_cam = self.transform(pc[:, :3])

            if label_percentages is not None:
                # Stratified sampling already done in get_pointcloud_from_arrays; no frustum cull.
                pc_arrays[index] = np.concatenate([xyz_cam, pc[:, 3:]], axis=1)
                continue

            _, frustum_mask = self.filter_frustum(xyz_cam)
            xyz_frustum     = xyz_cam[frustum_mask]
            extra_frustum   = pc[frustum_mask, 3:]
            result          = np.concatenate([xyz_frustum, extra_frustum], axis=1)

            # Ensure 7 columns: pad seg_id with zeros if segmentation was not run
            if result.shape[1] == 6:
                result = np.concatenate(
                    [result, np.zeros((len(result), 1), dtype=np.float32)], axis=1
                )

            # Uniformly decimate (or upsample by repetition) to exactly num_pts points
            if len(result) == 0:
                pc_arrays[index] = 0.0
            else:
                idx = np.round(np.linspace(0, len(result) - 1, num_pts)).astype(int)
                pc_arrays[index] = result[idx]

        return pc_arrays

    def process_live_ext(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        num_pts = 1024,
        label_percentages: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Process a single live RGBD frame into an external-frame point cloud.

        No coordinate transform, frustum culling, or ArUco detection.
        Uses self.segmentation when set; call seg.reset() before a new sequence.

        Args:
            color_rgb:          (H, W, 3) uint8 RGB — convert from RealSense BGR before calling.
            depth_m:            (H, W) float32 depth in metres, aligned to colour.
            num_pts:            Number of points to sample from the cloud.
            label_percentages:  dict mapping label_id -> percentage of num_pts, or None for
                                uniform sampling.  E.g. {1: 90, 2: 5, 3: 5} gives rope 90%,
                                gripper 5%, table 5%.  Each label is guaranteed its allocation
                                (upsampled by repetition if needed).

        Returns:
            (num_pts, 6) float32 xyzrgb        — when self.segmentation is None.
            (num_pts, 7) float32 xyzrgb+seg_id — when self.segmentation is set.
        """
        label_map = None
        if self.segmentation is not None:
            label_map, _ = self.segmentation.process_frame(color_rgb)

        return RGBDData.get_pointcloud_from_arrays(
            color_rgb, depth_m,
            self.tracker.camera_matrix,
            self.tracker.dist_coeffs,
            label_map=label_map,
            num_pts=num_pts,
            label_percentages=label_percentages,
        )

    def process_episode_ext(
        self,
        num_pts = 1024,
        label_percentages: dict[int, float] | None = None,
    ) -> np.ndarray | None:
        """Build an external-frame point cloud for every frame in the episode.

        No coordinate transform, frustum culling, or ArUco detection is
        performed — each frame's point cloud is uniformly decimated to exactly
        num_pts points.  Uses self.segmentation when set; call seg.reset() before
        starting a new episode.

        Args:
            num_pts:            Number of points to sample per frame.
            label_percentages:  dict mapping label_id -> percentage of num_pts, or None for
                                uniform sampling.  E.g. {1: 90, 2: 5, 3: 5} gives rope 90%,
                                gripper 5%, table 5%.  Each label is guaranteed its allocation
                                (upsampled by repetition if needed).

        Returns:
            (N, num_pts, 6) float32 xyzrgb        — when self.segmentation is None.
            (N, num_pts, 7) float32 xyzrgb+seg_id — when self.segmentation is set.
        """
        N = self.rgbd.num_frames
        n_cols = 7 if self.segmentation is not None else 6
        pc_arrays = np.zeros((N, num_pts, n_cols), dtype=np.float32)

        for index in range(N):
            color_rgb, depth_m = self.rgbd.get_frame(index)
            label_map = None
            if self.segmentation is not None:
                label_map, _ = self.segmentation.process_frame(color_rgb)
            pc = RGBDData.get_pointcloud_from_arrays(
                color_rgb, depth_m,
                self.tracker.camera_matrix, self.tracker.dist_coeffs,
                label_map=label_map, num_pts=num_pts,
                label_percentages=label_percentages,
            )
            pc_arrays[index] = pc

        return pc_arrays

    def save_processed_ext_no_seg(self, pc_arrays: np.ndarray) -> str:
        """Save external-frame (no-seg) episode data to a new zarr folder.

        The output folder has the same name as the input with
        '_processed_ext_no_seg' inserted before '.zarr'.  Actions,
        observations, and metadata are copied from the source episode as-is.

        Args:
            pc_arrays: (N, num_pts, 6) float32 array from process_episode_ext().

        Returns:
            Path to the written zarr folder.
        """
        src = self.rgbd.folder
        base = src[:-5] if src.endswith(".zarr") else src
        dst  = base + "_processed_ext_no_seg.zarr"

        src_root = zarr.open(src, mode='r')

        zarr.open_array(
            os.path.join(dst, "pointcloud"),
            mode='w', shape=pc_arrays.shape,
            dtype=np.float32, chunks=(1,) + pc_arrays.shape[1:],
        )[:] = pc_arrays

        for group_name in ("actions", "observations"):
            if group_name not in src_root:
                continue
            for key in src_root[group_name]:
                data = np.asarray(src_root[group_name][key])
                zarr.open_array(
                    os.path.join(dst, group_name, key),
                    mode='w', shape=data.shape,
                    dtype=data.dtype, chunks=(1,) + data.shape[1:],
                )[:] = data

        for fname in ("metadata.json", "metadata.npz"):
            src_file = os.path.join(src, fname)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(dst, fname))

        print(f"Saved: {dst}")
        return dst

    def save_processed_ext_seg(self, pc_arrays: np.ndarray) -> str:
        """Save external-frame (seg) episode data to a new zarr folder.

        The output folder has the same name as the input with
        '_processed_ext_seg' inserted before '.zarr'.  Actions,
        observations, and metadata are copied from the source episode as-is.

        Args:
            pc_arrays: (N, num_pts, 7) float32 array from process_episode_ext().

        Returns:
            Path to the written zarr folder.
        """
        src = self.rgbd.folder
        base = src[:-5] if src.endswith(".zarr") else src
        dst  = base + "_processed_ext_seg.zarr"

        src_root = zarr.open(src, mode='r')

        zarr.open_array(
            os.path.join(dst, "pointcloud"),
            mode='w', shape=pc_arrays.shape,
            dtype=np.float32, chunks=(1,) + pc_arrays.shape[1:],
        )[:] = pc_arrays

        for group_name in ("actions", "observations"):
            if group_name not in src_root:
                continue
            for key in src_root[group_name]:
                data = np.asarray(src_root[group_name][key])
                zarr.open_array(
                    os.path.join(dst, group_name, key),
                    mode='w', shape=data.shape,
                    dtype=data.dtype, chunks=(1,) + data.shape[1:],
                )[:] = data

        for fname in ("metadata.json", "metadata.npz"):
            src_file = os.path.join(src, fname)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(dst, fname))

        print(f"Saved: {dst}")
        return dst

    def save_processed(self, pc_arrays: np.ndarray) -> str:
        """Save processed episode data to a new zarr folder.

        The output folder has the same name as the input with '_processed'
        inserted before '.zarr'.  Images (RGB and depth) are not saved;
        the pointcloud array replaces them.  Actions, observations, and
        metadata are copied from the source episode as-is.

        Args:
            pc_arrays: (N, num_pts, 7) float32 array from process_episode().

        Returns:
            Path to the written zarr folder.
        """
        src = self.rgbd.folder
        base = src[:-5] if src.endswith(".zarr") else src
        dst  = base + "_processed.zarr"

        src_root = zarr.open(src, mode='r')

        # Pointcloud — one chunk per frame for sequential read access
        zarr.open_array(
            os.path.join(dst, "pointcloud"),
            mode='w', shape=pc_arrays.shape,
            dtype=np.float32, chunks=(1,) + pc_arrays.shape[1:],
        )[:] = pc_arrays

        # Copy actions and observations verbatim
        for group_name in ("actions", "observations"):
            if group_name not in src_root:
                continue
            for key in src_root[group_name]:
                data = np.asarray(src_root[group_name][key])
                zarr.open_array(
                    os.path.join(dst, group_name, key),
                    mode='w', shape=data.shape,
                    dtype=data.dtype, chunks=(1,) + data.shape[1:],
                )[:] = data

        # Copy metadata file(s)
        for fname in ("metadata.json", "metadata.npz"):
            src_file = os.path.join(src, fname)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(dst, fname))

        print(f"Saved: {dst}")
        return dst


def main():
    data_dir = "data"
    episodes = sorted(
        os.path.join(data_dir, e)
        for e in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, e))
        and not e.endswith("_processed.zarr")
        and (
            os.path.exists(os.path.join(data_dir, e, "metadata.json"))
            or os.path.exists(os.path.join(data_dir, e, "metadata.npz"))
        )
    )
    print(f"{len(episodes)} episodes found")
    seg = RealtimeSegmentation(reprompt_every=1)   # load GDINO + SAM 2 weights once for all episodes
    num_pts = 2048
    label_percentages = {1: 90.0, 2: 5.0, 3: 5.0}
    i = 1
    ext_fails = 0
    ego_fails = 0
    for episode in episodes:
        print(f"\n=== Processing {episode}: i = {i} ===")
        seg.reset()
        data     = RGBDData(episode)
        tracker  = ArucoTracker(data)
        pipeline = Ext2Ego(data, tracker, "config/camera.yaml", segmentation=seg)
        ret_ext = pipeline.process_episode_ext(num_pts, label_percentages)
        if ret_ext is None:
            ext_fails += 1
            print(f"  SKIP: episode {i} cannot be processed for ext.")
        else:
            print(f"ret_ext shape: {ret_ext.shape}")
            pipeline.save_processed_ext_seg(ret_ext)

        seg.reset()

        ret_ego = pipeline.process_episode(num_pts, label_percentages)
        if ret_ego is None:
            ego_fails += 1
            print(f"  SKIP: episode {i} cannot be processed for ego.")
        else:
            print(f"ret_ego shape: {ret_ego.shape}")
            pipeline.save_processed(ret_ego)
        
        i+=1
        print(f"ext fails: {ext_fails}")
        print(f"ego_fails: {ego_fails}")


def _save_pc_plot(pc: np.ndarray, path: str, title: str, use_seg_colors: bool = False) -> None:
    """Save a 3D scatter plot of a point cloud (N, 6) or (N, 7) to *path*."""
    xyz = pc[:, :3]
    if use_seg_colors and pc.shape[1] >= 7:
        seg_ids = pc[:, 6].astype(int)
        palette = {1: [1.0, 0.2, 0.2], 2: [0.2, 1.0, 0.2], 3: [0.2, 0.4, 1.0]}
        colors = np.array([palette.get(s, [0.6, 0.6, 0.6]) for s in seg_ids])
    else:
        colors = pc[:, 3:6] / 255.0 if pc.shape[1] >= 6 else None

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, linewidths=0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def step_by_step():
    """Execute process_episode step-by-step for frame 0 and save outputs at each stage."""
    out_dir = "step_by_step_output"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = "data"
    episodes = sorted(
        os.path.join(data_dir, e)
        for e in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, e))
        and not e.endswith("_processed.zarr")
        and (
            os.path.exists(os.path.join(data_dir, e, "metadata.json"))
            or os.path.exists(os.path.join(data_dir, e, "metadata.npz"))
        )
    )
    episode = episodes[0]
    print(f"Using episode: {episode}")

    num_pts = 2048
    label_percentages = {1: 90, 2: 5, 3: 5}
    index = 256

    seg = RealtimeSegmentation()
    data = RGBDData(episode)
    tracker = ArucoTracker(data)
    pipeline = Ext2Ego(data, tracker, "config/camera.yaml", segmentation=seg)

    # --- Step 1: Raw RGB frame ---
    color_rgb, depth_m = data.get_frame(index)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(color_rgb)
    ax.set_title("Step 1: Raw RGB frame")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step1_raw_rgb.png"), dpi=150)
    plt.close()
    print("  Saved: step1_raw_rgb.png")

    # --- Step 2: Depth map ---
    depth_valid = np.where(depth_m > 0, depth_m, np.nan)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(depth_valid, cmap="plasma")
    plt.colorbar(im, ax=ax, label="Depth (m)")
    ax.set_title("Step 2: Depth map")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step2_depth.png"), dpi=150)
    plt.close()
    print("  Saved: step2_depth.png")

    # --- Step 3: Segmentation label map ---
    label_map, _ = seg.process_frame(color_rgb)
    seg_palette = {1: [220, 50, 50], 2: [50, 220, 50], 3: [50, 100, 220]}
    seg_vis = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid, col in seg_palette.items():
        seg_vis[label_map == lid] = col
    blended = (color_rgb.astype(np.float32) * 0.5 + seg_vis.astype(np.float32) * 0.5).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(seg_vis)
    axes[0].set_title("Label map  (1=rope red, 2=gripper green, 3=table blue)")
    axes[0].axis("off")
    axes[1].imshow(blended)
    axes[1].set_title("Step 3: Segmentation overlay")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step3_segmentation.png"), dpi=150)
    plt.close()
    print("  Saved: step3_segmentation.png")

    # --- Step 4: Raw point cloud in RealSense frame ---
    # Stratified sampling (num_pts=2048, label_percentages) already applied here.
    pc = RGBDData.get_pointcloud_from_arrays(
        color_rgb, depth_m,
        tracker.camera_matrix, tracker.dist_coeffs,
        label_map=label_map,
        num_pts=num_pts,
        label_percentages=label_percentages,
    )
    np.save(os.path.join(out_dir, "step4_pc_realsense.npy"), pc)
    print("  Saved: step4_pc_realsense.npy")
    _save_pc_plot(pc, os.path.join(out_dir, "step4_pc_realsense_rgb.png"),
                  f"Step 4: Raw PC in RealSense frame — RGB colors ({len(pc)} pts)")
    _save_pc_plot(pc, os.path.join(out_dir, "step4_pc_realsense_seg.png"),
                  f"Step 4: Raw PC in RealSense frame — seg colors (1=rope, 2=gripper, 3=table)",
                  use_seg_colors=True)

    # --- Step 5: ArUco marker detection ---
    det = tracker.detect_plane(index)
    aruco_vis = color_rgb.copy()
    if det is not None:
        pts = np.round(det.corners).astype(int)
        for i in range(4):
            cv2.line(aruco_vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(aruco_vis, tuple(pt), 6, (0, 0, 255), -1)
        center = pts.mean(axis=0).astype(int)
        cv2.circle(aruco_vis, tuple(center), 6, (255, 0, 0), -1)
        print(f"  ArUco detected — tvec={det.tvec.round(4)}, angle={det.angle_deg:.1f}°")
    else:
        print("  Warning: ArUco not detected for frame 0")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(aruco_vis)
    ax.set_title("Step 5: ArUco marker detection  (green=outline, blue=corners, red=center)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step5_aruco_detection.png"), dpi=150)
    plt.close()
    print("  Saved: step5_aruco_detection.png")

    if det is None:
        print("Cannot continue past step 5: ArUco detection failed.")
        return

    # --- Step 6: Transform to ego camera frame ---
    # Since label_percentages is not None, process_episode skips frustum culling entirely —
    # the transform is the final step.
    R, t = tracker.get_camera_pose(det)
    pipeline.set_pose(R, t)
    xyz_cam = pipeline.transform(pc[:, :3])
    pc_ego = np.concatenate([xyz_cam, pc[:, 3:]], axis=1)
    np.save(os.path.join(out_dir, "step6_pc_ego.npy"), pc_ego)
    print("  Saved: step6_pc_ego.npy")
    _save_pc_plot(pc_ego, os.path.join(out_dir, "step6_pc_ego_rgb.png"),
                  f"Step 6: PC in ego camera frame — RGB colors ({len(pc_ego)} pts)")
    _save_pc_plot(pc_ego, os.path.join(out_dir, "step6_pc_ego_seg.png"),
                  f"Step 6: PC in ego camera frame — seg colors (1=rope, 2=gripper, 3=table)",
                  use_seg_colors=True)

    print(f"\nDone. All outputs saved to '{out_dir}/'")
    print(f"  Final point cloud shape: {pc_ego.shape}  (num_pts=2048, cols=xyzrgb+seg_id)")

def create_video():
    """Create a side-by-side RGB + segmentation video for every frame of every episode."""
    out_dir = "videos"
    os.makedirs(out_dir, exist_ok=True)

    data_dir = "data"
    episodes = sorted(
        os.path.join(data_dir, e)
        for e in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, e))
        and not e.endswith("_processed.zarr")
        and (
            os.path.exists(os.path.join(data_dir, e, "metadata.json"))
            or os.path.exists(os.path.join(data_dir, e, "metadata.npz"))
        )
    )
    print(f"{len(episodes)} episodes found")

    seg = RealtimeSegmentation(reprompt_every=1)
    seg_palette = {1: [220, 50, 50], 2: [50, 220, 50], 3: [50, 100, 220]}

    for episode in episodes:
        print(f"\nCreating video for: {episode}")
        seg.reset()
        data = RGBDData(episode)

        N   = data.num_frames
        fps = 1.0 / (data.timestamps[1] - data.timestamps[0]) if N > 1 else 30.0

        H, W    = data.get_frame(0)[0].shape[:2]
        ep_name = os.path.basename(episode.rstrip("/\\"))
        out_path = os.path.join(out_dir, f"{ep_name}_video.mp4")

        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H)
        )

        for i in range(N):
            color_rgb, _ = data.get_frame(i)
            label_map, _ = seg.process_frame(color_rgb)

            seg_vis = np.zeros((H, W, 3), dtype=np.uint8)
            for lid, col in seg_palette.items():
                seg_vis[label_map == lid] = col
            blended = (color_rgb.astype(np.float32) * 0.5 + seg_vis.astype(np.float32) * 0.5).astype(np.uint8)

            bgr_rgb = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            bgr_seg = cv2.cvtColor(blended,   cv2.COLOR_RGB2BGR)

            timestamp_text = f"frame {i}/{N - 1}   t={data.timestamps[i]:.2f}s"
            for img, label in ((bgr_rgb, "RGB"), (bgr_seg, "Segmentation (1=rope 2=gripper 3=table)")):
                cv2.putText(img, label,          (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,   0,   0  ), 3, cv2.LINE_AA)
                cv2.putText(img, label,          (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, timestamp_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,   0,   0  ), 3, cv2.LINE_AA)
                cv2.putText(img, timestamp_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(np.concatenate([bgr_rgb, bgr_seg], axis=1))

            if i % 20 == 0:
                print(f"  {i}/{N} frames")

        writer.release()
        print(f"Video saved: {out_path}")

if __name__ == "__main__":
    # main()
    # step_by_step()
    create_video()
