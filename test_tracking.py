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
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run in complete headless mode')
    
    return parser.parse_args()

def main():
    """Main test function"""
    args = parse_args()
    
    # Use the provided config file directly instead of creating a new one
    config_path = args.config
    
    # If input video is provided, override the config
    if args.input is not None and os.path.exists(args.input):
        print(f"Using input video file: {args.input}")
        # Create a temporary config to override the camera source
        with open(config_path, 'r') as f:
            config_data = f.read()
            
        # Update the video source
        temp_config_path = "temp_config.yml"
        # Look for the video_file entry and enable it
        if "video_file" in config_data:
            config_data = config_data.replace(
                "- enabled: false\n  name: video_file", 
                f"- enabled: true\n  name: video_file"
            )
            config_data = config_data.replace(
                'source: "samples/sample_video.mp4"', 
                f'source: "{args.input}"'
            )
        else:
            # Add a new video source if not found
            config_data += f"""
- enabled: true
  name: video_file
  source: "{args.input}"
"""
        
        with open(temp_config_path, 'w') as f:
            f.write(config_data)
            
        config_path = temp_config_path
    
    # Check camera configuration
    print(f"Using config file: {config_path}")
    
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
            
            # Guard against None values that might cause int() conversion errors
            frame_count_val = video.get(cv2.CAP_PROP_FRAME_COUNT)
            width_val = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height_val = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Convert to int with None checking
            frame_count = int(frame_count_val) if frame_count_val is not None else 0
            width = int(width_val) if width_val is not None else 0
            height = int(height_val) if height_val is not None else 0
            
            duration = frame_count / fps if fps > 0 else 0
            video.release()
            
            print(f"Video: {width}x{height}, {fps:.2f} FPS, {duration:.2f} seconds, {frame_count} frames")

if __name__ == "__main__":
    main() 