import matplotlib.pyplot as plt
import numpy as np
import yaml

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

    ground_size_mm = 10.0
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
        np.column_stack([
            np.full(yz_y.size, x_min, dtype=np.float32),
            yz_y.ravel(),
            yz_z.ravel(),
        ]),
        np.column_stack([
            np.full(yz_y.size, x_max, dtype=np.float32),
            yz_y.ravel(),
            yz_z.ravel(),
        ]),
        np.column_stack([
            xz_x.ravel(),
            np.full(xz_x.size, y_min, dtype=np.float32),
            xz_z.ravel(),
        ]),
        np.column_stack([
            xz_x.ravel(),
            np.full(xz_x.size, y_max, dtype=np.float32),
            xz_z.ravel(),
        ]),
        np.column_stack([
            xy_x.ravel(),
            xy_y.ravel(),
            np.full(xy_x.size, z_min, dtype=np.float32),
        ]),
        np.column_stack([
            xy_x.ravel(),
            xy_y.ravel(),
            np.full(xy_x.size, z_max, dtype=np.float32),
        ]),
    ]

    points = np.vstack(faces).astype(np.float32, copy=False)
    return np.unique(points, axis=0)

def plot_pc(pc: np.ndarray):
    """
    Display a 3D scatter plot for a point cloud.

    Args:
        pc: (N, 3) array-like collection of 3D points.

    Returns:
        None.
    """
    pc = np.asarray(pc)
    assert pc.ndim == 2 and pc.shape[1] == 3, "pc must have shape (N, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)

    if pc.shape[0] > 0:
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

def load_camera_config(path: str) -> dict:
    """
    Load camera parameters from a YAML configuration file.

    Args:
        path: Filesystem path to the camera config file.

    Returns:
        camera: Dict containing image size, field of view, and clip distances.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)["camera"]

    W = cfg["resolution"]["width"]
    H = cfg["resolution"]["height"]
    clip = cfg["clip"]

    fx = cfg["intrinsics"]["focal_length"]["fx"]
    fy = cfg["intrinsics"]["focal_length"]["fy"]
    hfov = np.degrees(2 * np.arctan(W / (2 * fx)))
    vfov = np.degrees(2 * np.arctan(H / (2 * fy)))

    return {
        "name":    cfg["name"],
        "width":   W,
        "height":  H,
        "hfov":    hfov,
        "vfov":    vfov,
        "z_near":  clip["near"],
        "z_far":   clip["far"],
    }

def transform_points_to_camera_frame(points_world: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    """
    Transform world-space points into camera-space.

    Args:
        points_world: (N, 3) float array of points in world frame.
        T_cw:         (4, 4) homogeneous world-to-camera transform.

    Returns:
        points_cam: (N, 3) float array of points in camera frame.
    """
    N = len(points_world)
    pts_h = np.hstack([points_world, np.ones((N, 1))])  # (N, 4) homogeneous
    pts_cam = (T_cw @ pts_h.T).T                         # (N, 4)
    return pts_cam[:, :3]


def build_frustum(hfov_deg: float, vfov_deg: float, z_near: float, z_far: float) -> dict:
    """
    Precompute the frustum parameters from camera intrinsics.

    Args:
        hfov_deg: Horizontal field of view in degrees.
        vfov_deg: Vertical field of view in degrees.
        z_near:   Near clip distance in meters.
        z_far:    Far clip distance in meters.

    Returns:
        A dict of precomputed frustum values ready for point testing.
    """
    return {
        "tan_h": np.tan(np.radians(hfov_deg / 2)),
        "tan_v": np.tan(np.radians(vfov_deg / 2)),
        "z_near": z_near,
        "z_far":  z_far,
    }


def filter_points_in_frustum(points_cam: np.ndarray, frustum: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove points that fall outside the camera frustum.

    Args:
        points_cam: (N, 3) float array of points in camera frame.
        frustum:    Dict produced by build_frustum().

    Returns:
        visible_points: (M, 3) subset of points_cam that lie inside the frustum.
        mask:           (N,) boolean array — True where a point passed all 6 tests.
    """
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    tan_h  = frustum["tan_h"]
    tan_v  = frustum["tan_v"]
    z_near = frustum["z_near"]
    z_far  = frustum["z_far"]

    mask = (
        (z > 0)          &   # point in front of camera
        (z >= z_near)    &   # near plane
        (z <= z_far)     &   # far plane
        (x <= z * tan_h) &   # right plane
        (x >= -z * tan_h) &  # left plane
        (y <= z * tan_v) &   # top plane
        (y >= -z * tan_v)    # bottom plane
    )

    return points_cam[mask], mask

def make_T_cw(R: np.ndarray, camera_pos_world: np.ndarray) -> np.ndarray:
    """
    Build a world-to-camera transform from a rotation matrix and
    the camera's position in world space.

    Args:
        R:                (3,3) rotation matrix (world-to-camera)
        camera_pos_world: (3,) camera position in world frame

    Returns:
        T_cw: (4,4) homogeneous world-to-camera transform
    """
    t = -R @ camera_pos_world  # translation in camera space
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

def main():
    """
    Run the example point-cloud and camera-config workflow.

    Args:
        None.

    Returns:
        None.
    """
    print("Hello from ext-to-ego!")
    pc = create_example_pc(100)
    pc = np.append(pc,create_box_pc(np.array([0, 0.2, 0.2]), 0.1, 0.1, 0.1, 50), axis = 0)
    print(pc.shape)
    plot_pc(pc)

    # Load Camera Intrinsics
    cam = load_camera_config("config/camera.yaml")
    print(cam)

    # Sim camera at (0, 0, 0.5) looking down.
    T_cw = make_T_cw(np.diag([1,-1,-1]), np.array([0,0,0.5]))

    # Transform points from initial frame to sim_camera frame
    pts_cam = transform_points_to_camera_frame(pc, T_cw)
    plot_pc(pts_cam)

    # Build frustum
    frustum = build_frustum(cam['hfov'], cam['vfov'], cam['z_near'], cam['z_far'])

    # Filter points out of the frustum
    pts_cam_filtered, mask = filter_points_in_frustum(pts_cam, frustum)

    # Plot out new ground
    plot_pc(pts_cam_filtered)
if __name__ == "__main__":
    main()
