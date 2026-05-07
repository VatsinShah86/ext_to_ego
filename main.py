import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
from scipy.spatial import cKDTree
from vision import RGBDData, ArucoTracker, MarkerDetection


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
    assert pc.ndim == 2 and pc.shape[1] in (3, 6), "pc must have shape (N, 3) or (N, 6)"

    xyz    = pc[:, :3]
    colors = pc[:, 3:] / 255.0 if pc.shape[1] == 6 else None

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

    def __init__(self, rgbd: RGBDData, tracker: ArucoTracker, config_path: str):
        self.rgbd    = rgbd
        self.tracker = tracker
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
        array in ego camera frame, and det is the MarkerDetection used to derive
        the ego camera pose (None if the marker was not found).
        """
        pc  = self.rgbd.get_pointcloud(index)   # (N, 6) xyzrgb
        det = self.tracker.detect_plane(index)
        self.tracker.plot_pointcloud_with_marker(index)
        if det is None:
            return np.empty((0, 6), dtype=np.float32), None

        R, t = self.tracker.get_camera_pose(det)
        self.set_pose(R, t)

        xyz_cam             = self.transform(pc[:, :3])
        _, frustum_mask     = self.filter_frustum(xyz_cam)

        xyz_frustum = xyz_cam[frustum_mask]
        rgb_frustum = pc[frustum_mask, 3:]
        result      = np.concatenate([xyz_frustum, rgb_frustum], axis=1)

        if occlusion:
            _, occ_mask = self.cull_occlusion(xyz_frustum)
            result      = result[occ_mask]

        return result, det


def main():
    print("Hello from ext-to-ego!")
    data = RGBDData('data/run_6_high_accuracy')
    tracker = ArucoTracker(data)
    pipeline = Ext2Ego(data, tracker, 'config/camera.yaml')
    pts, det = pipeline.process(1)
    plot_pc(pts)


if __name__ == "__main__":
    main()
