import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import SAM


class Segmentation:
    """SAM2-based image segmentation producing per-pixel integer label maps.

    All inputs and outputs are numpy arrays — no dependency on vision.py.
    Images are expected as (H, W, 3) uint8 RGB.
    """

    def __init__(
        self,
        model_path: str = "sam2.1_b.pt",
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ):
        self._model_path = model_path
        self._sam2_cfg   = sam2_cfg
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        # Ultralytics SAM2 — used for per-frame everything-mode segmentation
        self.model = SAM(model_path)
        self.model.to(self.device)
        # Meta SAM2 video predictor — loaded lazily on first segment_video() call
        self._video_predictor = None

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

    def segment_video(
        self, frames: np.ndarray, chunk_size: int = 500
    ) -> list[np.ndarray]:
        """Temporally consistent segmentation using the SAM2 video predictor.

        Processes the episode in fixed-size chunks to bound GPU/CPU memory use.
        Frame 0 of each chunk is seeded from the last label map of the previous
        chunk, so object IDs remain consistent across chunk boundaries.

        Args:
            frames:     (N, H, W, 3) uint8 RGB — all frames of the episode.
            chunk_size: Number of frames per SAM2 video-predictor call.

        Returns:
            List of N (H, W) int32 label maps.  IDs match those assigned by
            segment() on frame 0; background = 0.
        """
        import os
        import tempfile
        from PIL import Image

        if self._video_predictor is None:
            try:
                from sam2.build_sam import build_sam2_video_predictor
            except ImportError:
                raise ImportError(
                    "The 'sam2' package is required for video segmentation. "
                    "Install with: pip install sam2"
                )
            self._video_predictor = build_sam2_video_predictor(
                self._sam2_cfg, self._model_path, device=self.device
            )

        N, H, W = frames.shape[:3]
        label_maps = [np.zeros((H, W), dtype=np.int32) for _ in range(N)]

        # Seed from frame 0 using everything-mode (image model must be on GPU)
        seed_label_map = self.segment(frames[0])
        label_maps[0]  = seed_label_map

        # Move image model off GPU for the duration of video prediction
        self.model.to("cpu")
        torch.cuda.empty_cache()

        try:
            for chunk_start in range(0, N, chunk_size):
                chunk_end    = min(chunk_start + chunk_size, N)
                chunk_frames = frames[chunk_start:chunk_end]

                unique_ids = np.unique(seed_label_map)
                unique_ids = unique_ids[unique_ids > 0]

                print(f"  chunk [{chunk_start}, {chunk_end}) — "
                      f"{len(unique_ids)} objects")

                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, frame in enumerate(chunk_frames):
                        Image.fromarray(frame).save(
                            os.path.join(tmpdir, f"{i:05d}.jpg")
                        )

                    with torch.inference_mode():
                        state = self._video_predictor.init_state(
                            video_path=tmpdir, offload_state_to_cpu=True
                        )
                        for obj_id in unique_ids:
                            self._video_predictor.add_new_mask(
                                state, frame_idx=0, obj_id=int(obj_id),
                                mask=seed_label_map == obj_id,
                            )
                        for frame_idx, obj_ids, mask_logits in \
                                self._video_predictor.propagate_in_video(state):
                            masks = (mask_logits > 0.0).cpu().numpy()
                            lm = np.zeros((H, W), dtype=np.int32)
                            for obj_id, m in zip(obj_ids, masks):
                                lm[m[0]] = int(obj_id)
                            label_maps[chunk_start + frame_idx] = lm

                # Last frame of this chunk seeds the next chunk
                seed_label_map = label_maps[chunk_end - 1]
        except Exception as e:
            print(f"Ran into exception: {e}")
        finally:
            # Restore image model to GPU regardless of whether an error occurred
            self.model.to(self.device)

        return label_maps

    def segment_with_bbox(self, image_rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Segment using a bounding-box prompt — the primary SAM prompted mode.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            bbox:      (4,) array [x_min, y_min, x_max, y_max] in pixel coords.

        Returns:
            mask: (H, W) bool — highest-confidence SAM mask inside the bbox.
                  All-False if SAM returns nothing.
        """
        results = self.model(image_rgb, bboxes=[bbox.tolist()], verbose=False)
        masks_obj = results[0].masks
        if masks_obj is None:
            return np.zeros(image_rgb.shape[:2], dtype=bool)
        masks = masks_obj.data.cpu().numpy().astype(bool)  # (N, H, W), best-first
        if len(masks) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=bool)
        return masks[0]

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

    data = RGBDData("data/episode_20260507_232139.zarr")

    rgb, _ = data.get_frame(0)

    seg = Segmentation()
    label_map = seg.segment(rgb)
    seg.show(rgb, label_map)