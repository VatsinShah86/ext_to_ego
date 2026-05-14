"""
Text-prompted initialization (YOLOWorld) + SAM 2 temporal tracking.

Mirrors the architecture in realtime_segmentation_doc.md but substitutes
YOLOWorld for Grounding DINO — same open-vocabulary detection capability,
already present in the venv via ultralytics.

Label assignment (fixed):
    0  background
    1  rope
    2  robot gripper
    3  table surface

Outputs (H, W) int32 label maps compatible with:
    RGBDData.get_pointcloud(label_map=...)
    Ext2Ego.process_live(color_rgb, depth_m, label_map=...)

Two modes
---------
Live / streaming
    Call reset() before a new sequence, then process_frame() once per
    camera frame.  Uses SAM 2 image predictor with box prompts — truly
    frame-by-frame with no buffering required.  YOLOWorld re-prompts on
    confidence drop or at REPROMPT_EVERY-frame intervals; between reprompts
    the previous mask's bounding box is the SAM 2 prompt.

Episodic
    Call process_episode(rgbd) or process_episode_as_arrays(rgbd).
    Uses SAM 2 video predictor for full temporal memory within each chunk,
    with YOLOWorld re-prompting at chunk boundaries aligned to REPROMPT_EVERY.

Timing
    Every internal operation appends to self.timing (lists of ms values).
    Call timing_report() to print a formatted summary.
"""

import os
import tempfile
import time

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOWorld


CLASSES = ["rope", "robot gripper", "table surface"]
LABEL_IDS: dict[str, int] = {c: i + 1 for i, c in enumerate(CLASSES)}
# rope → 1 · robot gripper → 2 · table surface → 3

_LABEL_COLORS = {
    1: (0.95, 0.25, 0.25),   # rope       — red
    2: (0.25, 0.95, 0.35),   # gripper    — green
    3: (0.25, 0.45, 1.00),   # table      — blue
}
_LABEL_NAMES = {0: "background", 1: "rope", 2: "gripper", 3: "table"}

_DEFAULT_YOLOWORLD = "yolov8s-worldv2.pt"
_DEFAULT_SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_t.yaml"
_DEFAULT_SAM2_CKPT = "sam2.1_t.pt"


class RealtimeSegmentation:
    """Text-prompted init (YOLOWorld) + SAM 2 temporal tracking.

    Parameters
    ----------
    yoloworld_ckpt  : YOLOWorld model file (auto-downloaded if absent).
    sam2_cfg        : SAM 2 Hydra config name (relative to sam2 package).
    sam2_ckpt       : SAM 2 checkpoint path.
    box_threshold   : YOLOWorld confidence cutoff.  Lower → more detections.
    confidence_floor: SAM 2 IOU score below which a live-mode reprompt fires.
    reprompt_every  : Force YOLOWorld reprompt every N frames.
    device          : "cuda" / "cpu" (auto-detected if None).
    """

    def __init__(
        self,
        yoloworld_ckpt: str = _DEFAULT_YOLOWORLD,
        sam2_cfg: str = _DEFAULT_SAM2_CFG,
        sam2_ckpt: str = _DEFAULT_SAM2_CKPT,
        box_threshold: float = 0.35,
        confidence_floor: float = 0.70,
        reprompt_every: int = 150,
        device: str | None = None,
    ):
        self.box_threshold    = box_threshold
        self.confidence_floor = confidence_floor
        self.reprompt_every   = reprompt_every
        self.device           = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._sam2_cfg  = sam2_cfg
        self._sam2_ckpt = sam2_ckpt

        # YOLOWorld — open-vocab text → boxes, replaces Grounding DINO
        self._detector = YOLOWorld(yoloworld_ckpt)
        self._detector.set_classes(CLASSES)

        # Lazy-loaded SAM 2 predictors (loaded on first use)
        self._image_predictor = None   # SAM2ImagePredictor  (live mode)
        self._video_predictor = None   # SAM2VideoPredictor  (episodic mode)

        # Live-mode state
        self._frame_idx: int = 0
        self._masks_by_label: dict[str, np.ndarray] = {}
        self._scores_by_label: dict[str, float]     = {}

        # Timing accumulators (ms)
        self.timing: dict[str, list[float]] = {
            "detect_ms":      [],   # one entry per YOLOWorld call
            "sam2_image_ms":  [],   # one entry per process_frame() call
            "sam2_video_ms":  [],   # one entry per _propagate_chunk() call
            "frame_total_ms": [],   # end-to-end per process_frame() call
        }

    # ── Lazy model loaders ─────────────────────────────────────────────────

    def _get_image_predictor(self):
        if self._image_predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(self._sam2_cfg, self._sam2_ckpt, device=self.device)
            self._image_predictor = SAM2ImagePredictor(model)
        return self._image_predictor

    def _get_video_predictor(self):
        if self._video_predictor is None:
            from sam2.build_sam import build_sam2_video_predictor
            self._video_predictor = build_sam2_video_predictor(
                self._sam2_cfg, self._sam2_ckpt, device=self.device
            )
        return self._video_predictor

    # ── Detection ──────────────────────────────────────────────────────────

    def _detect_boxes(self, frame_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Run YOLOWorld; return the highest-confidence box per class.

        Returns {label: (4,) float32 xyxy}.  Missing key = class not found.
        Appends to self.timing["detect_ms"].
        """
        t0 = time.perf_counter()

        results   = self._detector.predict(frame_rgb, verbose=False, conf=self.box_threshold)
        boxes     = results[0].boxes.xyxy.cpu().numpy()
        scores    = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        best: dict[str, tuple[np.ndarray, float]] = {}
        for box, score, cid in zip(boxes, scores, class_ids):
            if cid >= len(CLASSES):
                continue
            label = CLASSES[cid]
            if label not in best or score > best[label][1]:
                best[label] = (box, float(score))

        self.timing["detect_ms"].append((time.perf_counter() - t0) * 1e3)
        return {label: v[0] for label, v in best.items()}

    # ── SAM 2 image predictor (live mode) ──────────────────────────────────

    def _sam2_predict_boxes(
        self,
        frame_rgb: np.ndarray,
        boxes_by_label: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Predict one mask per label using SAM 2 image predictor + box prompt.

        Returns (masks_by_label, scores_by_label) where masks are (H, W) bool
        and scores are SAM 2 IOU estimates in [0, 1].
        Appends total time for all labels to self.timing["sam2_image_ms"].
        """
        predictor = self._get_image_predictor()
        t0 = time.perf_counter()

        with torch.inference_mode():
            predictor.set_image(frame_rgb)

        masks_out:  dict[str, np.ndarray] = {}
        scores_out: dict[str, float]      = {}

        for label, box in boxes_by_label.items():
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            masks_out[label]  = masks[0].astype(bool)
            scores_out[label] = float(scores[0])

        self.timing["sam2_image_ms"].append((time.perf_counter() - t0) * 1e3)
        return masks_out, scores_out

    # ── Shared helpers ─────────────────────────────────────────────────────

    def _masks_to_label_map(
        self,
        masks_by_label: dict[str, np.ndarray],
        frame_shape: tuple,
    ) -> np.ndarray:
        """Combine per-object masks into a single (H, W) int32 label map.

        Written in priority order: table first (large background), then gripper,
        then rope last — so thin/small objects are never overwritten by larger ones.
        """
        H, W = frame_shape[:2]
        lm = np.zeros((H, W), dtype=np.int32)
        for label in ["table surface", "robot gripper", "rope"]:
            if label in masks_by_label:
                lm[masks_by_label[label]] = LABEL_IDS[label]
        return lm

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> np.ndarray | None:
        """(H, W) bool → [x_min, y_min, x_max, y_max] xyxy, or None if empty."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

    # ── Timing helpers ──────────────────────────────────────────────────────

    def clear_timing(self) -> None:
        """Clear all accumulated timing data."""
        for v in self.timing.values():
            v.clear()

    def timing_report(self) -> None:
        """Print a formatted timing summary to stdout."""
        def _fmt(vals: list[float], label: str) -> str:
            if not vals:
                return f"  {label:<30s}  no data"
            return (
                f"  {label:<30s}"
                f"  mean={np.mean(vals):6.1f} ms"
                f"  min={np.min(vals):6.1f} ms"
                f"  max={np.max(vals):6.1f} ms"
                f"  n={len(vals)}"
            )

        det   = self.timing["detect_ms"]
        img   = self.timing["sam2_image_ms"]
        vid   = self.timing["sam2_video_ms"]
        total = self.timing["frame_total_ms"]

        print("\n─── Timing report ───────────────────────────────────────────────")
        print(_fmt(det,   "YOLOWorld detection"))
        print(_fmt(img,   "SAM 2 image predict (live)"))
        if vid:
            per_frame = [v / c for v, c in zip(vid, self._chunk_sizes)]
            print(_fmt(vid,       "SAM 2 video propagation (chunk)"))
            print(_fmt(per_frame, "  ↳ per frame"))
        print(_fmt(total, "Live mode end-to-end"))
        if total:
            fps = 1000.0 / np.mean(total)
            print(f"  {'Effective live FPS':<30s}  {fps:.1f} Hz")
        print("─────────────────────────────────────────────────────────────────")

    # ── Live / streaming mode ──────────────────────────────────────────────

    def reset(self) -> None:
        """Reset streaming state.  Call before starting a new sequence."""
        self._frame_idx      = 0
        self._masks_by_label  = {}
        self._scores_by_label = {}

    def process_frame(
        self,
        frame_rgb: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Process one frame for live / streaming use.

        Args:
            frame_rgb: (H, W, 3) uint8 RGB.

        Returns:
            label_map : (H, W) int32 — 0 bg · 1 rope · 2 gripper · 3 table.
            scores    : {label: float} — SAM 2 IOU scores for detected objects.
        """
        t_frame = time.perf_counter()

        needs_reprompt = (
            self._frame_idx == 0
            or self._frame_idx % self.reprompt_every == 0
            or not self._masks_by_label
            or any(s < self.confidence_floor for s in self._scores_by_label.values())
        )

        if needs_reprompt:
            boxes_by_label = self._detect_boxes(frame_rgb)
        else:
            # Derive SAM 2 prompts from previous masks' bounding boxes
            boxes_by_label = {}
            for label, mask in self._masks_by_label.items():
                box = self._mask_to_box(mask)
                if box is not None:
                    boxes_by_label[label] = box

        if boxes_by_label:
            self._masks_by_label, self._scores_by_label = self._sam2_predict_boxes(
                frame_rgb, boxes_by_label
            )

        self._frame_idx += 1
        self.timing["frame_total_ms"].append((time.perf_counter() - t_frame) * 1e3)
        label_map = self._masks_to_label_map(self._masks_by_label, frame_rgb.shape)
        return label_map, dict(self._scores_by_label)

    # ── Episodic mode ──────────────────────────────────────────────────────

    def _propagate_chunk(
        self,
        frames: list[np.ndarray],
        seed_masks: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Propagate masks through one chunk using the SAM 2 video predictor.

        Writes frames to a temporary JPEG directory, seeds object memories
        from seed_masks on frame 0, then propagates forward.
        Appends chunk wall-time to self.timing["sam2_video_ms"].

        Args:
            frames:     List of (H, W, 3) uint8 RGB frames.
            seed_masks: {label: (H, W) bool} for frame 0 of this chunk.

        Returns:
            List of N (H, W) int32 label maps, one per frame in the chunk.
        """
        predictor = self._get_video_predictor()
        N = len(frames)
        H, W = frames[0].shape[:2]
        label_maps = [np.zeros((H, W), dtype=np.int32) for _ in range(N)]

        if not seed_masks:
            return label_maps

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(tmpdir, f"{i:05d}.jpg"))

            with torch.inference_mode():
                state = predictor.init_state(
                    video_path=tmpdir, offload_state_to_cpu=True
                )
                for label, mask in seed_masks.items():
                    predictor.add_new_mask(
                        state, frame_idx=0,
                        obj_id=LABEL_IDS[label], mask=mask,
                    )
                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                    lm    = np.zeros((H, W), dtype=np.int32)
                    masks = (mask_logits > 0.0).cpu().numpy()
                    for obj_id, m in zip(obj_ids, masks):
                        lm[m[0]] = int(obj_id)
                    label_maps[frame_idx] = lm

        self.timing["sam2_video_ms"].append((time.perf_counter() - t0) * 1e3)
        if not hasattr(self, "_chunk_sizes"):
            self._chunk_sizes = []
        self._chunk_sizes.append(N)
        return label_maps

    def process_episode(
        self,
        rgbd,
        chunk_size: int = 100,
    ) -> list[np.ndarray]:
        """Process all frames of an episode with SAM 2 temporal propagation.

        YOLOWorld initializes frame 0 and re-prompts at every REPROMPT_EVERY-
        aligned chunk boundary.  Between reprompts, the last frame of each
        chunk seeds the next, preserving SAM 2 temporal memory across chunks.

        Args:
            rgbd:       RGBDData instance — must expose get_frame(i) → (rgb, depth)
                        and the num_frames property.
            chunk_size: Frames per SAM 2 video-predictor call.  Trade-off:
                        smaller = less GPU RAM; larger = better temporal memory.

        Returns:
            List of N (H, W) int32 label maps, one per episode frame.
        """
        N = rgbd.num_frames
        all_maps: list[np.ndarray] = []
        seed_masks: dict[str, np.ndarray] = {}
        self._chunk_sizes = []

        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            frames    = [rgbd.get_frame(i)[0] for i in range(chunk_start, chunk_end)]

            needs_reprompt = (
                chunk_start == 0
                or chunk_start % self.reprompt_every == 0
                or not seed_masks
            )

            if needs_reprompt:
                boxes = self._detect_boxes(frames[0])
                if boxes:
                    seed_masks, _ = self._sam2_predict_boxes(frames[0], boxes)
                    print(f"  [reprompt] frame {chunk_start}: {list(seed_masks.keys())}")
                else:
                    print(f"  [reprompt] frame {chunk_start}: nothing detected — keeping previous seed")

            print(f"  chunk [{chunk_start}, {chunk_end}): {len(seed_masks)} objects")
            chunk_maps = self._propagate_chunk(frames, seed_masks)
            all_maps.extend(chunk_maps)

            # Seed next chunk from the last propagated label map
            last_lm    = chunk_maps[-1]
            seed_masks = {
                label: (last_lm == lid).astype(bool)
                for label, lid in LABEL_IDS.items()
                if (last_lm == lid).any()
            }

        return all_maps

    def process_episode_as_arrays(
        self,
        rgbd,
        chunk_size: int = 100,
    ) -> np.ndarray:
        """process_episode() stacked into a (N, H, W) int32 array."""
        return np.stack(self.process_episode(rgbd, chunk_size=chunk_size), axis=0)


# ── Visualisation helpers ──────────────────────────────────────────────────

def _overlay_masks(rgb: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Return a float RGB image with semi-transparent per-class mask overlays."""
    out = rgb.astype(float) / 255.0
    for lid, color in _LABEL_COLORS.items():
        mask = label_map == lid
        if not mask.any():
            continue
        for c, v in enumerate(color):
            out[..., c] = np.where(mask, (1 - alpha) * out[..., c] + alpha * v, out[..., c])
    return out


def _make_legend_patches():
    import matplotlib.patches as mpatches
    return [
        mpatches.Patch(color=_LABEL_COLORS[lid], label=_LABEL_NAMES[lid])
        for lid in sorted(_LABEL_COLORS)
    ]


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    from vision import RGBDData

    # ── Locate first raw episode ────────────────────────────────────────────
    episodes = sorted(
        e for e in glob.glob("data/episode_*.zarr")
        if not e.endswith("_processed.zarr")
        and not e.endswith("_processed_ext_no_seg.zarr")
    )
    if not episodes:
        raise SystemExit("No episodes found in data/")

    ep_path = episodes[0]
    print(f"\nEpisode : {ep_path}")
    rgbd = RGBDData(ep_path)
    print(f"Frames  : {rgbd.num_frames}")

    # ── Build segmentor (lower threshold so we're more likely to detect all objects) ──
    seg = RealtimeSegmentation(box_threshold=0.25)

    # ── Warmup: load models before timing ──────────────────────────────────
    print("\nWarming up models…")
    _warmup_frame, _ = rgbd.get_frame(0)
    seg.process_frame(_warmup_frame)
    seg.reset()
    seg.clear_timing()

    # ── Live-mode pass: process first TEST_N frames with full timing ────────
    TEST_N = min(120, rgbd.num_frames)
    print(f"\nRunning live mode on frames 0–{TEST_N - 1}…")

    label_maps: list[np.ndarray] = []
    all_scores: list[dict[str, float]] = []

    for i in range(TEST_N):
        frame_rgb, _ = rgbd.get_frame(i)
        lm, scores   = seg.process_frame(frame_rgb)
        label_maps.append(lm)
        all_scores.append(scores)

    # ── Timing report ───────────────────────────────────────────────────────
    seg.timing_report()

    # ── Find best frames (most objects visible) ─────────────────────────────
    def _object_count(lm: np.ndarray) -> int:
        return sum(1 for lid in LABEL_IDS.values() if (lm == lid).any())

    def _labels_found(lm: np.ndarray) -> list[str]:
        return [_LABEL_NAMES[lid] for lid in sorted(LABEL_IDS.values()) if (lm == lid).any()]

    # Sort frames: prefer more objects, break ties by total mask area
    ranked = sorted(
        range(TEST_N),
        key=lambda i: (
            _object_count(label_maps[i]),
            int((label_maps[i] > 0).sum()),
        ),
        reverse=True,
    )

    # Pick up to 4 diverse frames that together cover all 3 objects
    shown: list[int] = []
    covered: set[int] = set()
    for idx in ranked:
        lm = label_maps[idx]
        new_ids = {lid for lid in LABEL_IDS.values() if (lm == lid).any()} - covered
        if new_ids or (not shown):
            shown.append(idx)
            covered |= new_ids
        if len(shown) == 4 and covered == set(LABEL_IDS.values()):
            break
    # Fall back: just take the top-4 ranked frames if we didn't find ideal ones
    if len(shown) < 4:
        shown = ranked[:4]
    shown.sort()

    print(f"\nSelected frames: {shown}")
    for i in shown:
        print(f"  frame {i:03d}: {_labels_found(label_maps[i])}  scores={all_scores[i]}")

    # ── Figure 1: per-object mask breakdown ────────────────────────────────
    # Layout: one row per selected frame, columns = [combined | rope | gripper | table]
    COL_LABELS = ["combined", "rope", "gripper", "table"]
    COL_IDS    = [None, 1, 2, 3]          # None = all objects, else the label ID
    n_rows, n_cols = len(shown), len(COL_LABELS)

    fig1, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for row, idx in enumerate(shown):
        frame_rgb, _ = rgbd.get_frame(idx)
        lm           = label_maps[idx]
        scores       = all_scores[idx]
        found        = _labels_found(lm)

        for col, (col_label, lid) in enumerate(zip(COL_LABELS, COL_IDS)):
            ax = axes[row, col]

            if lid is None:
                # Combined: all three overlaid on RGB
                img = _overlay_masks(frame_rgb, lm)
                ax.imshow(img)
                ax.set_title(f"frame {idx}\n{found}", fontsize=8)
            else:
                # Individual mask: object pixels in its colour, rest dark grey
                mask   = lm == lid
                color  = np.array(_LABEL_COLORS[lid])
                canvas = np.full((*frame_rgb.shape[:2], 3), 0.15)  # dark background
                base   = frame_rgb.astype(float) / 255.0
                # blend: 55% original + 45% object colour inside mask
                for c in range(3):
                    canvas[..., c] = np.where(
                        mask,
                        0.55 * base[..., c] + 0.45 * color[c],
                        0.15,
                    )
                score_txt = f"{scores.get(CLASSES[lid - 1], 0):.2f}" if scores.get(CLASSES[lid - 1]) else "—"
                present   = "✓" if mask.any() else "✗"
                ax.imshow(canvas)
                ax.set_title(
                    f"{col_label}  {present}\nIOU {score_txt}",
                    fontsize=8,
                    color=_LABEL_COLORS[lid],
                )

            ax.axis("off")

    fig1.suptitle(
        "Segmentation: combined + per-object masks\n"
        "(rope = red · gripper = green · table = blue)",
        fontsize=11,
    )
    plt.tight_layout()
    fig1.savefig("segmentation_overlays.png", dpi=120, bbox_inches="tight")
    print("\nSaved: segmentation_overlays.png")

    # ── Figure 2: timing breakdown ──────────────────────────────────────────
    det   = seg.timing["detect_ms"]
    img   = seg.timing["sam2_image_ms"]
    total = seg.timing["frame_total_ms"]

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.suptitle("Per-frame timing (live mode)", fontsize=12)

    def _hist(ax, data, title, color, budget_ms=None):
        ax.hist(data, bins=20, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(data), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean {np.mean(data):.1f} ms")
        if budget_ms:
            ax.axvline(budget_ms, color="red", linestyle=":", linewidth=1.2,
                       label=f"budget {budget_ms} ms")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ms")
        ax.set_ylabel("frames")
        ax.legend(fontsize=8)

    _hist(axes2[0], det,   "YOLOWorld detection\n(reprompt frames only)", "#e07b54")
    _hist(axes2[1], img,   "SAM 2 image predict\n(live, per frame)",      "#5b8dd9")
    _hist(axes2[2], total, "End-to-end per frame\n(live mode)",            "#6dbf67", budget_ms=66)

    plt.tight_layout()
    fig2.savefig("segmentation_timing.png", dpi=120, bbox_inches="tight")
    print("Saved: segmentation_timing.png")

    plt.show()
