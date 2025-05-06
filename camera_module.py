import cv2
import threading
import time
import os


class CameraSource:
    def __init__(self, name, source, enabled=True):
        """
        Initialize a camera source.
        
        Args:
            name (str): Unique name for the camera
            source (int/str): Camera source (0 for webcam, path for video files, URL for IP cameras)
            enabled (bool): Whether the camera is enabled
        """
        self.name = name
        self.source = source
        self.enabled = enabled
        self.cap = None
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps = 0
        self.last_frame_time = 0
        
        # For video files
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)
        self.total_frames = 0
        self.current_frame_index = 0
        self.loop_video = True  # Set to True to loop video files
        
    def start(self):
        """Start capturing from this camera source"""
        if not self.enabled:
            print(f"Camera '{self.name}' is disabled in config")
            return False
            
        try:
            # Check if source is a video file that exists
            if isinstance(self.source, str) and not self.source.isdigit():
                if os.path.isfile(self.source):
                    print(f"Opening video file: {self.source}")
                    self.is_video_file = True
                else:
                    print(f"Video file not found: {self.source}, checking if it's a camera/stream URL")
            
            # Try to open the source
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(f"Failed to open camera/video '{self.name}' (source: {self.source})")
                return False
            
            # Get video file properties if applicable
            if self.is_video_file:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Video file has {self.total_frames} frames")
                
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Error starting camera '{self.name}': {e}")
            return False
    
    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        while self.running:
            ret, frame = self.cap.read()
            current_time = time.time()
            
            # Handle end of video file if needed
            if not ret and self.is_video_file:
                if self.loop_video:
                    print(f"End of video file reached, restarting: {self.source}")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_index = 0
                    ret, frame = self.cap.read()
                else:
                    print(f"End of video file reached: {self.source}")
                    self.running = False
                    break
            
            # Update frame counter for video files
            if self.is_video_file and ret:
                self.current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Calculate FPS
            if self.last_frame_time != 0:
                self.fps = 1 / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            
            with self.lock:
                self.ret = ret
                self.frame = frame
                
            # Avoid maxing out CPU and control playback speed for video files
            if self.is_video_file:
                # Get video's natural FPS and adjust sleep time accordingly
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if video_fps > 0:
                    sleep_time = 1.0 / video_fps
                    time.sleep(max(0.001, sleep_time))  # Ensure minimum sleep time
                else:
                    time.sleep(0.03)  # Default to ~30 FPS if can't determine
            else:
                time.sleep(0.01)  # Regular camera/stream
    
    def get_frame(self):
        """Get the latest frame from the camera"""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
    
    def get_progress(self):
        """Get progress information for video files"""
        if not self.is_video_file or self.total_frames == 0:
            return None
            
        return {
            'current_frame': self.current_frame_index,
            'total_frames': self.total_frames,
            'progress_percent': (self.current_frame_index / self.total_frames) * 100
        }
    
    def stop(self):
        """Stop capturing from this camera source"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CameraModule:
    def __init__(self, camera_configs):
        """
        Initialize the camera module with multiple camera sources.
        
        Args:
            camera_configs (list): List of camera configuration dictionaries
        """
        self.cameras = {}
        self.active_camera = None
        
        # Initialize cameras from config
        for cam_config in camera_configs:
            name = cam_config.get('name')
            source = cam_config.get('source')
            enabled = cam_config.get('enabled', True)
            
            if name and source is not None:
                # Convert numeric string sources to integers (for webcams)
                if isinstance(source, str) and source.isdigit():
                    source = int(source)
                    
                self.cameras[name] = CameraSource(name, source, enabled)
    
    def start_all_cameras(self):
        """Start all enabled cameras"""
        started_cameras = []
        for name, camera in self.cameras.items():
            if camera.start():
                started_cameras.append(name)
                # Set the first successfully started camera as active
                if self.active_camera is None:
                    self.active_camera = name
        
        return started_cameras
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        for camera in self.cameras.values():
            camera.stop()
    
    def get_camera_frame(self, camera_name=None):
        """
        Get frame from the specified camera or active camera.
        
        Args:
            camera_name (str, optional): Name of the camera to get frame from.
                                        If None, uses the active camera.
        
        Returns:
            tuple: (success, frame)
        """
        if camera_name is None:
            camera_name = self.active_camera
            
        if camera_name not in self.cameras:
            return False, None
            
        return self.cameras[camera_name].get_frame()
    
    def get_camera_progress(self, camera_name=None):
        """
        Get progress information for a video file.
        
        Args:
            camera_name (str, optional): Name of the camera to get progress from.
                                        If None, uses the active camera.
        
        Returns:
            dict: Progress information or None if not a video file
        """
        if camera_name is None:
            camera_name = self.active_camera
            
        if camera_name not in self.cameras:
            return None
            
        return self.cameras[camera_name].get_progress()
    
    def set_active_camera(self, camera_name):
        """
        Set the active camera.
        
        Args:
            camera_name (str): Name of the camera to set as active
            
        Returns:
            bool: Success or failure
        """
        if camera_name in self.cameras:
            self.active_camera = camera_name
            return True
        return False
    
    def get_active_camera_name(self):
        """Get the name of the currently active camera"""
        return self.active_camera
    
    def get_camera_list(self):
        """Get list of all available cameras"""
        return list(self.cameras.keys()) 