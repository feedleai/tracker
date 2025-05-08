#!/usr/bin/env python3
import argparse
import cv2
import time
import os
from face_tracker_app import FaceTrackerApp

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Enhanced Person Tracking System')
    parser.add_argument('--config', type=str, default='config.yml', 
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input video file (optional)')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Path to output video file')
    parser.add_argument('--show_fps', action='store_true',
                        help='Show FPS in output video')
    parser.add_argument('--webcam', type=int, default=0,
                        help='Webcam ID to use if no input file is provided')
    parser.add_argument('--test_mode', type=str, choices=['all', 'face', 'gait', 'pose', 'occlusion'],
                        default='all', help='Test specific module')
    
    return parser.parse_args()

def create_test_config(args):
    """Create a test-specific configuration based on args"""
    # Check if the config file exists
    if not os.path.exists(args.config):
        print(f"Config file '{args.config}' not found. Creating default config.")
        # Create basic config file
        with open(args.config, 'w') as f:
            f.write("""
cameras:
  sources:
    - name: Default Camera
      source: 0
      type: webcam
  default_camera: Default Camera

face_recognition:
  model: buffalo_l
  detection_threshold: 0.5
  recognition_threshold: 0.5
  use_gpu: true

person_tracking:
  model_weights: models/yolov5m.pt
  confidence_threshold: 0.4
  track_buffer: 30
  match_threshold: 0.8
  use_gpu: true

gait_recognition:
  sequence_length: 20
  min_track_length: 10
  similarity_threshold: 0.7
  smoothing_window: 7
  feature_dim: 128
  db_path: gait_database.pkl

pose_estimation:
  model_path: models/pose
  confidence_threshold: 0.5
  use_gpu: true
  max_history: 30

fusion:
  face_weight: 0.6
  gait_weight: 0.25
  pose_weight: 0.15
  appearance_weight: 0.2
  motion_weight: 0.1
  fusion_threshold: 0.55
  adaptive_weights: true
  time_window: 5.0

occlusion:
  overlap_threshold: 0.5
  min_area_ratio: 0.3
  max_history: 20

database:
  use_db: true
  db_path: face_database.db

output:
  show_video: true
  show_person_detection: true
  show_person_id: true
  show_pose: true
  show_gait: true
  show_fusion: true
  show_occlusions: true
  save_detections: false
  output_dir: ./detected_faces
""")

    # If input video is provided, modify camera config
    if args.input is not None:
        # Ensure the input file exists
        if not os.path.exists(args.input):
            print(f"Input file '{args.input}' not found.")
            return

        # Create a temporary config with the input video
        temp_config_path = "temp_config.yml"
        with open(args.config, 'r') as f:
            config_data = f.read()
            
        # Replace camera section
        config_data = config_data.replace(
            "cameras:",
            f"""cameras:
  sources:
    - name: Test Video
      source: {args.input}
      type: video
  default_camera: Test Video"""
        )
        
        with open(temp_config_path, 'w') as f:
            f.write(config_data)
            
        return temp_config_path
        
    return args.config

def main():
    """Main test function"""
    args = parse_args()
    
    # Create test configuration
    config_path = create_test_config(args)
    
    # Initialize tracker app
    app = FaceTrackerApp(config_path)
    
    # Set output path
    app.output_video_path = args.output
    
    # Check if specific modules should be disabled for testing
    if args.test_mode != 'all':
        if args.test_mode != 'face':
            app.gait_module = None
            app.pose_module = None
            app.fusion_module = None
            app.occlusion_handler = None
            print("Testing FACE recognition only")
        elif args.test_mode == 'gait':
            app.pose_module = None
            app.fusion_module = None
            app.occlusion_handler = None
            print("Testing FACE + GAIT recognition")
        elif args.test_mode == 'pose':
            app.gait_module = None
            app.fusion_module = None
            app.occlusion_handler = None
            print("Testing FACE + POSE recognition")
        elif args.test_mode == 'occlusion':
            app.gait_module = None
            app.fusion_module = None
            print("Testing FACE + OCCLUSION handling")
    
    # Start the application
    try:
        app.start()
    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Clean up temporary config if created
        if config_path != args.config and os.path.exists(config_path):
            os.remove(config_path)
            
        # If output video was created, show info
        if os.path.exists(app.output_video_path):
            print(f"Output video saved to: {app.output_video_path}")
            
            # Get video information
            video = cv2.VideoCapture(app.output_video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            video.release()
            
            print(f"Video: {width}x{height}, {fps:.2f} FPS, {duration:.2f} seconds, {frame_count} frames")

if __name__ == "__main__":
    main() 