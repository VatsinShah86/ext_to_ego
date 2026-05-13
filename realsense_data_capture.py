import pyrealsense2 as rs
import numpy as np
import time
import zarr

class DepthCamera:
    def __init__(self, resolution_width, resolution_height):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        # Get depth scale of the device
        self.depth_scale =  depth_sensor.get_depth_scale()
            # Create an align object
        align_to = rs.stream.color

        self.align = rs.align(align_to)
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("device product line:", device_product_line)
        config.enable_stream(rs.stream.depth,  resolution_width,  resolution_height, rs.format.z16, 6)
        config.enable_stream(rs.stream.color,  resolution_width,  resolution_height, rs.format.bgr8, 30)
        
        # Start streaming
        profile = self.pipeline.start(config)
        
        # Set depth sensor to high accuracy preset for better quality
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)  # 3 = High Accuracy preset
        print("RealSense configured for high accuracy mode")
        
        color_sensor = profile.get_device().first_color_sensor()

        # Disable auto-exposure first — required before setting manual value
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)

        # Set exposure in microseconds — lower = less blur, darker image
        # D435i color range: ~1 to 10000 microseconds
        color_sensor.set_option(rs.option.exposure, 500)  # start here, tune down if still blurry

        # Compensate for darker image by boosting gain
        # Range: 0-128, higher = brighter but more noise
        color_sensor.set_option(rs.option.gain, 32)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
        self.temporal.set_option(rs.option.filter_smooth_delta, 40)

        self.hole_filling = rs.hole_filling_filter()
        # Warmup: read frames until we get valid non-empty point clouds
        print("Warming up camera...")
        
        time.sleep(10)

    def get_raw_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None
        
        # Apply filter pipeline to depth only
        # depth_frame = self.decimation.process(depth_frame)
        # depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        return True, depth_frame, color_frame
    
    def get_depth_scale(self):
        """
        "scaling factor" refers to the relation between depth map units and meters; 
        it has nothing to do with the focal length of the camera.
        Depth maps are typically stored in 16-bit unsigned integers at millimeter scale, thus to obtain Z value in meters, the depth map pixels need to be divided by 1000.
        """
        return self.depth_scale

    def release(self):
        self.pipeline.stop()

    @staticmethod
    def depth2PointCloud(depth, rgb, depth_scale, clip_distance_max, num_points=1024):
    
        intrinsics = depth.profile.as_video_stream_profile().intrinsics
        depth_raw = np.asanyarray(depth.get_data())
        depth = depth_raw.astype(np.float32) * depth_scale  # 1000 mm => 0.001 meters
        rgb = np.asanyarray(rgb.get_data())
        rows, cols = depth.shape

        # DEBUG: Print depth statistics
        valid_depth_raw = depth_raw[depth_raw > 0]
        if len(valid_depth_raw) > 0:
            print(f"[DEBUG] Raw depth - min: {valid_depth_raw.min()}, max: {valid_depth_raw.max()}, mean: {valid_depth_raw.mean():.1f}")
            print(f"[DEBUG] Scaled depth (meters) - min: {depth[depth > 0].min():.3f}, max: {depth[depth > 0].max():.3f}, mean: {depth[depth > 0].mean():.3f}")
        else:
            print("[DEBUG] WARNING: No valid depth pixels found!")
        print(f"[DEBUG] clip_distance_max: {clip_distance_max} meters")

        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        r = r.astype(float)
        c = c.astype(float)

        valid = (depth > 0) & (depth < clip_distance_max)  # remove from the depth image all values above a given value (meters).
        valid = np.ravel(valid)
        num_valid = np.sum(valid)
        print(f"[DEBUG] Valid points after filtering: {num_valid} / {len(valid)}")
        
        z = depth 
        x = z * (c - intrinsics.ppx) / intrinsics.fx
        y = z * (r - intrinsics.ppy) / intrinsics.fy
    
        z = np.ravel(z)[valid]
        x = np.ravel(x)[valid]
        y = np.ravel(y)[valid]
        
        # Extract BGR channels (color is in BGR format, not RGB)
        b_val = np.ravel(rgb[:,:,0])[valid]
        g_val = np.ravel(rgb[:,:,1])[valid]
        r_val = np.ravel(rgb[:,:,2])[valid]
        
        # Store as BGR to preserve original channel ordering
        pointsxyzrgb = np.dstack((x, y, z, b_val, g_val, r_val))
        pointsxyzrgb = pointsxyzrgb.reshape(-1, 6).astype(np.float32)
        
        # Decimate to requested number of points
        total_points = len(pointsxyzrgb)
        if num_points < total_points:
            indices = np.linspace(0, total_points - 1, num_points, dtype=int)
            pointsxyzrgb = pointsxyzrgb[indices]

        return pointsxyzrgb
    

def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    camera = DepthCamera(1280,720)
    try:
        # Read one frame for point cloud generation (warmup already done in __init__)
        success, depth_frame, color_frame = camera.get_raw_frame()
        if not success:
            print("Failed to read frame")
        else:
            # Generate point cloud with decimation
            point_cloud = DepthCamera.depth2PointCloud(
                depth_frame, 
                color_frame, 
                camera.get_depth_scale(), 
                clip_distance_max=3.0,
                num_points=4096*16
            )
            
            print(f"Point cloud shape: {point_cloud.shape}")
            
            # Extract XYZ and RGB
            xyz = point_cloud[:, :3]
            rgb = point_cloud[:, 3:6]
            
            # Plot point cloud
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Convert BGR to RGB for proper visualization
            # Point cloud stores [B, G, R], swap to [R, G, B]
            rgb_corrected = rgb[:, [2, 1, 0]]  # BGR -> RGB
            rgb_normalized = rgb_corrected / 255.0
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb_normalized, s=2, alpha=0.6)
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')
            ax.set_title(f'RealSense Point Cloud ({len(xyz):,} points)')
            
            plt.tight_layout()
            print("Displaying point cloud visualization...")
            plt.show()
    finally:
        camera.release()

def record_data(duration_seconds=10, output_dir=None, max_frames=None):
    '''
    Record raw camera data without any post-processing.
    
    Args:
        duration_seconds: How long to record (seconds). Default 10s.
        output_dir: Where to save data. Default: /tmp/raw_camera_data/
        max_frames: Max number of frames to record. If None, record until duration expires.
    
    Saves:
        - depth_raw.npy: Raw depth frames (N, 480, 640) uint16
        - color_raw.npy: Raw color frames (N, 480, 640, 3) uint8
        - metadata.npz: Intrinsics, depth_scale, timestamps, frame_count
    '''
    import json
    import os
    from datetime import datetime
    
    if output_dir is None:
        output_dir = "/tmp/raw_camera_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    camera = DepthCamera(640, 480)
    
    depth_frames = []
    color_frames = []
    timestamps = []
    
    print(f"\n{'='*60}")
    print(f"Recording raw camera data for {duration_seconds} seconds...")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    try:
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            if max_frames and frame_count >= max_frames:
                break
            
            success, depth_frame, color_frame = camera.get_raw_frame()
            if not success:
                print(f"Frame {frame_count}: Failed to read frame")
                continue
            
            # Extract raw data - NO processing
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            color_raw = np.asanyarray(color_frame.get_data()).astype(np.uint8)
            
            depth_frames.append(depth_raw)
            color_frames.append(color_raw)
            timestamps.append(time.time() - start_time)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frame {frame_count}: depth shape {depth_raw.shape}, "
                      f"color shape {color_raw.shape}, FPS: {fps:.1f}")
        
        print(f"\nRecording complete: {frame_count} frames captured\n")
        
        # Convert to numpy arrays
        depth_array = np.array(depth_frames, dtype=np.uint16)  # (N, H, W)
        color_array = np.array(color_frames, dtype=np.uint8)   # (N, H, W, 3)
        
        # Get frames to retrieve camera profiles and intrinsics
        _, depth_frame, color_frame = camera.get_raw_frame()
        depth_profile = depth_frame.profile.as_video_stream_profile()
        color_profile = color_frame.profile.as_video_stream_profile()
        depth_intrinsics = depth_profile.intrinsics
        color_intrinsics = color_profile.intrinsics
        
        # Get extrinsics (transformation from depth to color frame)
        depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)
        
        # Save raw depth as zarr
        depth_path = os.path.join(output_dir, "depth_raw.zarr")
        zarr.open_array(depth_path, mode='w', shape=depth_array.shape, dtype=depth_array.dtype, chunks=(1, 480, 640))[:] = depth_array
        print(f"✓ Saved {depth_array.nbytes / 1e9:.2f}GB depth data: {depth_path}")
        print(f"  Shape: {depth_array.shape}, dtype: {depth_array.dtype}")
        
        # Save raw color as zarr
        color_path = os.path.join(output_dir, "color_raw.zarr")
        zarr.open_array(color_path, mode='w', shape=color_array.shape, dtype=color_array.dtype, chunks=(1, 480, 640, 3))[:] = color_array
        print(f"✓ Saved {color_array.nbytes / 1e9:.2f}GB color data: {color_path}")
        print(f"  Shape: {color_array.shape}, dtype: {color_array.dtype}")
        
        # Prepare comprehensive metadata
        metadata = {
            'frame_count': frame_count,
            'depth_scale': float(camera.get_depth_scale()),
            'timestamps': timestamps,
            'depth_intrinsics': {
                'fx': depth_intrinsics.fx,
                'fy': depth_intrinsics.fy,
                'ppx': depth_intrinsics.ppx,
                'ppy': depth_intrinsics.ppy,
                'width': depth_intrinsics.width,
                'height': depth_intrinsics.height,
                'model': str(depth_intrinsics.distortion_model) if hasattr(depth_intrinsics, 'distortion_model') else 'None',
                'coeffs': list(depth_intrinsics.coeffs) if hasattr(depth_intrinsics, 'coeffs') else [],
            },
            'color_intrinsics': {
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'ppx': color_intrinsics.ppx,
                'ppy': color_intrinsics.ppy,
                'width': color_intrinsics.width,
                'height': color_intrinsics.height,
                'model': str(color_intrinsics.distortion_model) if hasattr(color_intrinsics, 'distortion_model') else 'None',
                'coeffs': list(color_intrinsics.coeffs) if hasattr(color_intrinsics, 'coeffs') else [],
            },
            'depth_to_color_extrinsics': {
                'rotation': np.array(depth_to_color_extrinsics.rotation).tolist() if depth_to_color_extrinsics else None,
                'translation': np.array(depth_to_color_extrinsics.translation).tolist() if depth_to_color_extrinsics else None,
            },
            'depth_shape': depth_array.shape,
            'color_shape': color_array.shape,
            'color_format': 'BGR',  # Colors are stored as BGR
            'recording_time': datetime.now().isoformat(),
        }
        
        metadata_path = os.path.join(output_dir, "metadata.npz")
        np.savez(metadata_path, **metadata)
        print(f"✓ Saved metadata: {metadata_path}")
        print(f"  Depth scale: {metadata['depth_scale']}")
        print(f"  Depth intrinsics: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}, "
              f"ppx={depth_intrinsics.ppx:.2f}, ppy={depth_intrinsics.ppy:.2f}")
        print(f"  Color intrinsics: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}, "
              f"ppx={color_intrinsics.ppx:.2f}, ppy={color_intrinsics.ppy:.2f}")
        if depth_to_color_extrinsics:
            print(f"  Depth-to-Color translation: {depth_to_color_extrinsics.translation}")
        
        print(f"\n{'='*60}")
        print(f"All raw data saved successfully.")
        print(f"{'='*60}\n")
        
    finally:
        camera.release()

def validate_recording(output_dir):
    """
    Load and validate the recorded raw data.
    """
    import os
    depth_path = os.path.join(output_dir, "depth_raw.zarr")
    color_path = os.path.join(output_dir, "color_raw.zarr")
    metadata_path = os.path.join(output_dir, "metadata.npz")
    
    # Load data from zarr
    depth_array = zarr.open_array(depth_path, mode='r')[:]
    color_array = zarr.open_array(color_path, mode='r')[:]
    metadata = np.load(metadata_path, allow_pickle=True)
    
    print(f"Loaded depth data: shape {depth_array.shape}, dtype {depth_array.dtype}")
    print(f"Loaded color data: shape {color_array.shape}, dtype {color_array.dtype}")
    print(f"Loaded metadata: {metadata.files}")
    
    # Basic validation
    assert depth_array.ndim == 3 and depth_array.dtype == np.uint16, "Depth array has unexpected shape or dtype"
    assert color_array.ndim == 4 and color_array.dtype == np.uint8, "Color array has unexpected shape or dtype"
    
    frame_count = metadata['frame_count'].item()
    assert depth_array.shape[0] == frame_count, "Depth frame count does not match metadata"
    assert color_array.shape[0] == frame_count, "Color frame count does not match metadata"
    
    print("Validation successful: raw data is consistent with metadata.")

    # visualize 10 frames of depth and color side-by-side for sanity check
    import matplotlib.pyplot as plt
    num_visualize = min(10, frame_count)

    # Get evenly spaced frame indices across entire recording (0 to frame_count)
    frame_indices = np.linspace(0, frame_count - 1, num_visualize, dtype=int)
    
    # Note: zarr arrays support lazy loading, so slicing is efficient
    # Create num_visualize number of figures with depth and color side by side
    for i in frame_indices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(depth_array[i], cmap='gray')
        axes[0].set_title(f"Depth Frame {i}")
        axes[0].axis('off')
        
        # Convert BGR to RGB for proper visualization
        color_rgb = color_array[i, :, :, [2, 1, 0]]  # BGR -> RGB
        axes[1].imshow(color_rgb)
        axes[1].set_title(f"Color Frame {i} (BGR->RGB)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        # Usage: python depth_tmp.py record [duration_seconds] [output_dir]
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        record_data(duration_seconds=duration, output_dir=output_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Usage: python depth_tmp.py validate [output_dir]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        if output_dir is None:
            print("Please provide the output directory to validate, e.g. python depth_tmp.py validate /tmp/raw_camera_data")
        else:
            validate_recording(output_dir)
    else:
        main()
