"""Exercise every non-saving Ext2Ego method against a real episode."""

import matplotlib
matplotlib.use("Agg")   # prevent plt.show() from blocking

import sys, os
os.chdir("/home/vatsin/ext_to_ego")

import numpy as np
from vision import RGBDData, ArucoTracker
from realtime_segmentation import RealtimeSegmentation
from main import Ext2Ego

EPISODE = "data/episode_20260513_181516.zarr"
CONFIG  = "config/camera.yaml"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

results = []

def check(name, value, expected_shape=None):
    if value is None:
        tag = WARN
        detail = "returned None"
    elif hasattr(value, "shape"):
        if expected_shape is not None and value.shape != expected_shape:
            tag    = FAIL
            detail = f"shape={value.shape}  expected={expected_shape}"
        else:
            tag    = PASS
            detail = f"shape={value.shape}"
    elif isinstance(value, tuple):
        tag    = PASS
        detail = f"tuple len={len(value)}"
    else:
        tag    = PASS
        detail = str(type(value).__name__)
    results.append((tag, name, detail))
    print(f"  [{tag}]  {name:45s}  {detail}")


# ── Lightweight RGBDData wrapper for slicing episodes ─────────────────────────
class SlicedRGBD:
    """Presents a prefix of an RGBDData as a smaller episode."""
    def __init__(self, rgbd: RGBDData, n: int):
        self._rgbd      = rgbd
        self.num_frames = n
        self.folder     = rgbd.folder
    def get_frame(self, i):
        return self._rgbd.get_frame(i)


# ── Setup ─────────────────────────────────────────────────────────────────────
print("\n=== Setup ===")
rgbd    = RGBDData(EPISODE)
tracker = ArucoTracker(rgbd)
seg     = RealtimeSegmentation()
print(f"  Episode frames : {rgbd.num_frames}")

pipeline = Ext2Ego(rgbd, tracker, CONFIG, segmentation=seg)
check("Ext2Ego.__init__", pipeline.T_cw if False else "ok")

# ── Find a frame with a good ArUco detection ──────────────────────────────────
print("\n=== Finding ArUco frame ===")
det0, good_idx = None, 0
for i in range(min(50, rgbd.num_frames)):
    d = tracker.detect_plane(i)
    if d is not None:
        det0, good_idx = d, i
        print(f"  ArUco found at frame {i}")
        break
if det0 is None:
    print("  WARNING: no ArUco found in first 50 frames — some tests will be skipped")

color_rgb, depth_m = rgbd.get_frame(good_idx)

# ── set_pose ──────────────────────────────────────────────────────────────────
print("\n=== set_pose ===")
if det0 is not None:
    R, t = tracker.get_camera_pose(det0)
    pipeline.set_pose(R, t)
    check("set_pose → T_cw shape", pipeline.T_cw, (4, 4))
else:
    # set a dummy pose so downstream tests can proceed
    pipeline.set_pose(np.eye(3), np.zeros(3))
    print("  (dummy pose set)")

# ── transform ─────────────────────────────────────────────────────────────────
print("\n=== transform ===")
dummy_pts = np.random.default_rng(0).random((200, 3)).astype(np.float32)
result_transform = pipeline.transform(dummy_pts)
check("transform", result_transform, (200, 3))

# ── filter_frustum ────────────────────────────────────────────────────────────
print("\n=== filter_frustum ===")
pts_frust, mask_frust = pipeline.filter_frustum(result_transform)
check("filter_frustum visible pts", pts_frust)
check("filter_frustum mask",        mask_frust, (200,))
print(f"  {mask_frust.sum()} / 200 points pass frustum")

# ── cull_occlusion ────────────────────────────────────────────────────────────
print("\n=== cull_occlusion ===")
if len(pts_frust):
    pts_occ, mask_occ = pipeline.cull_occlusion(pts_frust)
    check("cull_occlusion visible pts", pts_occ)
    check("cull_occlusion mask",        mask_occ, (len(pts_frust),))
    print(f"  {mask_occ.sum()} / {len(pts_frust)} pass occlusion")
else:
    print("  SKIP — no frustum points to cull")

# ── cull_occlusion_soft ───────────────────────────────────────────────────────
print("\n=== cull_occlusion_soft ===")
if len(pts_frust):
    pts_soft, mask_soft = pipeline.cull_occlusion_soft(pts_frust, pixel_radius=12.5)
    check("cull_occlusion_soft visible pts", pts_soft)
    check("cull_occlusion_soft mask",        mask_soft, (len(pts_frust),))
    print(f"  {mask_soft.sum()} / {len(pts_frust)} pass soft occlusion")
else:
    print("  SKIP — no frustum points")

# ── process ───────────────────────────────────────────────────────────────────
print("\n=== process ===")
seg.reset()
pc_proc, det_proc = pipeline.process(good_idx, occlusion=False)
check("process(occlusion=False) pc",  pc_proc)
check("process(occlusion=False) det", np.array([1]) if det_proc is not None else None)

seg.reset()
pc_proc2, det_proc2 = pipeline.process(good_idx, occlusion=True)
check("process(occlusion=True) pc",   pc_proc2)

# ── detect_from_sam ───────────────────────────────────────────────────────────
print("\n=== detect_from_sam ===")
if det0 is not None:
    result_dsam = pipeline.detect_from_sam(good_idx, det0)
    check("detect_from_sam", np.array([1]) if result_dsam is not None else None)
    print(f"  returned MarkerDetection: {result_dsam is not None}")
else:
    print("  SKIP — no ArUco det available")

# ── detect_from_sam_frame ─────────────────────────────────────────────────────
print("\n=== detect_from_sam_frame ===")
if det0 is not None:
    result_dsamf = pipeline.detect_from_sam_frame(color_rgb, depth_m, det0)
    check("detect_from_sam_frame", np.array([1]) if result_dsamf is not None else None)
    print(f"  returned MarkerDetection: {result_dsamf is not None}")
else:
    print("  SKIP — no ArUco det available")

# ── process_live ──────────────────────────────────────────────────────────────
print("\n=== process_live ===")
seg.reset()
pc_live, det_live = pipeline.process_live(color_rgb, depth_m)
check("process_live (no prev_det) pc",  pc_live,  (1024, 7))
check("process_live (no prev_det) det", np.array([1]) if det_live is not None else None)
print(f"  ArUco detected: {det_live is not None}")

color_rgb2, depth_m2 = rgbd.get_frame(min(good_idx + 1, rgbd.num_frames - 1))
pc_live2, det_live2 = pipeline.process_live(color_rgb2, depth_m2, prev_det=det_live)
check("process_live (with prev_det) pc", pc_live2, (1024, 7))

# ── process_live_ext ──────────────────────────────────────────────────────────
print("\n=== process_live_ext ===")
seg.reset()
pc_ext = pipeline.process_live_ext(color_rgb, depth_m)
check("process_live_ext", pc_ext, (1024, 7))
print(f"  non-zero points: {(pc_ext[:, :3] != 0).any(axis=1).sum()}")

# ── process_episode_ext (first 15 frames) ────────────────────────────────────
print("\n=== process_episode_ext (15 frames) ===")
seg.reset()
small_rgbd   = SlicedRGBD(rgbd, 15)
pipeline_ext = Ext2Ego(small_rgbd, tracker, CONFIG, segmentation=seg)
ret_ext = pipeline_ext.process_episode_ext()
check("process_episode_ext", ret_ext, (15, 1024, 7))

# ── process_episode (first 15 frames) ────────────────────────────────────────
print("\n=== process_episode (15 frames) ===")
seg.reset()
pipeline_ep = Ext2Ego(small_rgbd, tracker, CONFIG, segmentation=seg)
ret_ep = pipeline_ep.process_episode()
if ret_ep is None:
    check("process_episode", None)
else:
    check("process_episode", ret_ep, (15, 1024, 7))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for t, *_ in results if t == PASS)
warned = sum(1 for t, *_ in results if t == WARN)
failed = sum(1 for t, *_ in results if t == FAIL)
print(f"Results:  {passed} passed  |  {warned} warnings  |  {failed} failed")
if failed:
    print("\nFailed checks:")
    for tag, name, detail in results:
        if tag == FAIL:
            print(f"  {name}: {detail}")
print("=" * 60)
