import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import zarr
from scipy.spatial import cKDTree
from vision import RGBDData, ArucoTracker, MarkerDetection
from segmentation import Segmentation


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
                 segmentation: Segmentation | None = None):
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
        rgb, _    = self.rgbd.get_frame(index)
        label_map = self.segmentation.segment(rgb) if self.segmentation is not None else None

        pc  = self.rgbd.get_pointcloud(index, label_map=label_map)  # (N,6) or (N,7)
        det = self.tracker.detect_plane(index)

        self.rgbd.plot_pointcloud(index, label_map = label_map)
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
        rgb, _ = self.rgbd.get_frame(index)
        corners = prev_det.corners                          # (4, 2) float32
        pad = 5.0
        bbox = np.array([
            corners[:, 0].min() - pad,
            corners[:, 1].min() - pad,
            corners[:, 0].max() + pad,
            corners[:, 1].max() + pad,
        ])
        mask = self.segmentation.segment_with_bbox(rgb, bbox)
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
        mask = self.segmentation.segment_with_bbox(color_rgb, bbox)
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
    ) -> tuple[np.ndarray, MarkerDetection | None]:
        """Process a single live RGBD frame into a (1024, 7) ego-frame point cloud.

        Args:
            color_rgb:  (H, W, 3) uint8 RGB — convert from RealSense BGR before calling.
            depth_m:    (H, W) float32 depth in metres, aligned to color.
            prev_det:   Last good MarkerDetection; enables SAM fallback when ArUco is occluded.
            label_map:  (H, W) int32 segmentation labels, or None to run per-frame SAM2.
            occlusion:  Whether to apply occlusion culling (slower).

        Returns:
            (pc, det) — pc is (1024, 7) float32 in ego frame; det is the MarkerDetection
            used this frame (pass back as prev_det on the next call).
            Returns (zeros, None) if marker detection fails completely.
        """
        if label_map is None and self.segmentation is not None:
            label_map = self.segmentation.segment(color_rgb)

        pc  = RGBDData.get_pointcloud_from_arrays(
            color_rgb, depth_m,
            self.tracker.camera_matrix,
            self.tracker.dist_coeffs,
            label_map=label_map,
        )

        det = self.tracker.detect_plane_from_frame(color_rgb, depth_m)

        if det is not None:
            prev_det = det
        elif prev_det is not None:
            det = self.detect_from_sam_frame(color_rgb, depth_m, prev_det)
            if det is not None:
                prev_det = det

        if det is None:
            return np.zeros((1024, 7), dtype=np.float32), None

        R, t = self.tracker.get_camera_pose(det)
        self.set_pose(R, t)

        xyz_cam         = self.transform(pc[:, :3])
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
            return np.zeros((1024, 7), dtype=np.float32), det

        idx = np.round(np.linspace(0, len(result) - 1, 1024)).astype(int)
        return result[idx], det

    def process_episode(self) -> np.ndarray | None:
        N         = self.rgbd.num_frames
        pc_arrays = np.zeros((N, 1024, 7), dtype=np.float32)

        prev_det = None
        for index in range(N):
            # Per-frame segmentation — same model and approach used at runtime.
            if self.segmentation is not None:
                rgb, _ = self.rgbd.get_frame(index)
                label_map = self.segmentation.segment(rgb)
            else:
                label_map = None

            pc  = self.rgbd.get_pointcloud(index, label_map=label_map)  # (N,6) or (N,7)
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

            xyz_cam         = self.transform(pc[:, :3])
            _, frustum_mask = self.filter_frustum(xyz_cam)

            xyz_frustum   = xyz_cam[frustum_mask]
            extra_frustum = pc[frustum_mask, 3:]          # rgb (+ seg_id if present)
            result        = np.concatenate([xyz_frustum, extra_frustum], axis=1)

            # Ensure 7 columns: pad seg_id with zeros if segmentation was not run
            if result.shape[1] == 6:
                result = np.concatenate(
                    [result, np.zeros((len(result), 1), dtype=np.float32)], axis=1
                )

            # Uniformly decimate (or upsample by repetition) to exactly 1024 points
            if len(result) == 0:
                pc_arrays[index] = 0.0
            else:
                idx = np.round(np.linspace(0, len(result) - 1, 1024)).astype(int)
                pc_arrays[index] = result[idx]

        return pc_arrays

    def process_episode_ext(self, segmentation: Segmentation | None = None) -> np.ndarray | None:
        """Build a (N, 1024, 6) external-frame RGB point cloud for every frame.

        No coordinate transform, frustum culling, or ArUco detection is
        performed — each frame's full scene point cloud is uniformly decimated
        to exactly 1024 points and stored as xyzrgb.

        Args:
            segmentation: Reserved for future use; passing a value raises
                          NotImplementedError.

        Returns:
            (N, 1024, 6) float32 xyzrgb array, or None on failure.
        """
        if segmentation is not None:
            raise NotImplementedError("Segmentation support is not yet implemented for process_episode_ext.")

        N = self.rgbd.num_frames
        pc_arrays = np.zeros((N, 1024, 6), dtype=np.float32)

        for index in range(N):
            pc = self.rgbd.get_pointcloud(index, label_map=None, num_pts=1024)  # (1024, 6) xyzrgb
            pc_arrays[index] = pc[:, :6]

        return pc_arrays

    def save_processed_ext_no_seg(self, pc_arrays: np.ndarray) -> str:
        """Save external-frame (no-seg) episode data to a new zarr folder.

        The output folder has the same name as the input with
        '_processed_ext_no_seg' inserted before '.zarr'.  Actions,
        observations, and metadata are copied from the source episode as-is.

        Args:
            pc_arrays: (N, 1024, 6) float32 array from process_episode_ext().

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
            dtype=np.float32, chunks=(1, 1024, 6),
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
            pc_arrays: (N, 1024, 7) float32 array from process_episode().

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
            dtype=np.float32, chunks=(1, 1024, 7),
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

    # seg = Segmentation(
    #     model_path="mobile_sam.pt",  # MobileSAM — matches policy_runtime
    #     points_per_side=16,
    #     points_per_batch=128,
    #     pred_iou_thresh=0.88,
    #     stability_score_thresh=0.95,
    #     min_mask_region_area=100,
    #     crop_n_layers=0,
    # )

    for episode in episodes:
        print(f"\n=== Processing {episode} ===")
        data     = RGBDData(episode)
        tracker  = ArucoTracker(data)
        pipeline = Ext2Ego(data, tracker, "config/camera.yaml")
        ret = pipeline.process_episode_ext()
        if ret is None:
            print(f"  SKIP: episode cannot be processed.")
        else:
            pipeline.save_processed_ext_no_seg(ret)

if __name__ == "__main__":
    main()
