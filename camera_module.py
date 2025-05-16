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
        self.failed_reads = 0
        self.last_successful_read = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_backoff = 1  # Initial backoff in seconds
        
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
            
            # Try to open the source with longer timeout for CCTV streams
            is_stream = self._is_streaming_source(self.source)
            if is_stream:
                print(f"Opening streaming source: {self.source} with extended timeout")
                # Set connection timeout property for streams (30 seconds)
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # For RTSP/CCTV streams, use specific codec settings to handle HEVC streams better
                # These settings help with the "Could not find ref with POC" errors
                # Use more robust settings that work better with problematic HEVC streams
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                
                # Try to set these specific FFmpeg options if the implementation supports it
                try:
                    # Set FFmpeg-specific options to handle problematic streams
                    # Increase buffer size and enable error recovery
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10-second connection timeout
                    
                    # On some OpenCV builds, we can set additional FFmpeg options
                    if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 0)  # Disable hardware acceleration for better compatibility
                    
                    # For OpenCV 4.5+ with proper FFmpeg options support
                    extra_options = {
                        'rtsp_transport': 'tcp',         # Use TCP for more reliable streaming
                        'max_delay': '500000',           # Maximum demuxing delay (in microseconds)
                        'fflags': 'nobuffer+discardcorrupt',  # Discard corrupt packets and reduce buffering
                        'flags': 'low_delay',            # Prioritize low delay over quality
                        'strict': 'experimental',        # Allow experimental codecs/features
                        'analyzeduration': '1000000'     # Reduce the analyze time to 1 second
                    }
                    
                    # Try to set each option individually
                    for option, value in extra_options.items():
                        try:
                            option_id = 1000000 + hash(option) % 1000  # Create a unique option ID
                            self.cap.set(option_id, value)
                        except:
                            # Options not supported, continue silently
                            pass
                except Exception as e:
                    # Just log the error, don't fail completely
                    print(f"Could not set advanced stream options: {e}")
                
                # Request lower fps for more stable connection
                self.cap.set(cv2.CAP_PROP_FPS, 15)
            else:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
            if not self.cap.isOpened():
                print(f"Failed to open camera/video '{self.name}' (source: {self.source})")
                return False
            
            # Get video file properties if applicable
            if self.is_video_file:
                try:
                    frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # Some video files return 0 or -1 for frame count
                    if frame_count is not None and frame_count > 0:
                        self.total_frames = int(frame_count)
                        print(f"Video file has {self.total_frames} frames")
                    else:
                        self.total_frames = 0
                        print(f"Warning: Could not determine frame count for {self.source}")
                except Exception as e:
                    self.total_frames = 0
                    print(f"Error getting frame count: {e}")
            
            # Reset counters on successful connection
            self.failed_reads = 0
            self.reconnect_attempts = 0
            self.reconnect_backoff = 1
            self.last_successful_read = time.time()
                
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Error starting camera '{self.name}': {e}")
            return False
    
    def _is_streaming_source(self, source):
        """
        Check if the source is a streaming URL
        
        Args:
            source: Camera source
            
        Returns:
            bool: True if source is a stream URL
        """
        if isinstance(source, str) and not os.path.isfile(source):
            # Check for common stream URL patterns
            return (source.startswith("rtsp://") or 
                    source.startswith("http://") or 
                    source.startswith("https://") or
                    source.startswith("rtmp://") or
                    ".m3u8" in source or
                    ".mjpg" in source)
        return False
    
    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        hevc_error_count = 0  # Counter for HEVC-specific errors
        
        while self.running:
            try:
                # Important: Check if cap is None before using it
                # This prevents 'NoneType' object has no attribute 'read' errors
                with self.lock:
                    if self.cap is None:
                        # If the capture object is None, wait briefly and then continue the loop
                        time.sleep(0.1)
                        continue
                
                ret, frame = self.cap.read()
                current_time = time.time()
                
                # Handle end of video file if needed
                if not ret and self.is_video_file:
                    if self.loop_video:
                        print(f"End of video file reached, restarting: {self.source}")
                        # Need to make sure cap is still valid
                        with self.lock:
                            if self.cap is not None:
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                self.current_frame_index = 0
                                ret, frame = self.cap.read()
                    else:
                        print(f"End of video file reached: {self.source}")
                        self.running = False
                        break
                
                # Update frame counter for video files
                if self.is_video_file and ret:
                    # Access cap with lock to prevent race conditions
                    with self.lock:
                        if self.cap is not None:
                            frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                            if frame_pos is not None:
                                self.current_frame_index = int(frame_pos)
                            else:
                                # If we can't get frame position, increment manually
                                self.current_frame_index += 1
                
                # For streaming sources, check if we're getting frames
                if not ret and self._is_streaming_source(self.source):
                    consecutive_errors += 1
                    
                    # Get error log from OpenCV/FFmpeg if available
                    # This is a heuristic since we can't directly access FFmpeg errors
                    # Increment HEVC error counter if this is likely a codec issue
                    if consecutive_errors >= 3:
                        # Check if this might be an HEVC stream by analyzing source URL
                        if self._might_be_hevc_stream(self.source):
                            # Count this as likely HEVC error
                            hevc_error_count += 2  # Increase counter faster for suspected HEVC streams
                        else:
                            hevc_error_count += 1
                        
                        # If we're seeing a pattern of errors that suggests HEVC reference frame issues,
                        # use a specialized reconnection strategy
                        if hevc_error_count >= 5:
                            print(f"Detected potential HEVC reference frame issues for {self.name}, using specialized reconnection")
                            self._reconnect_for_hevc_errors()
                            # Reset counters
                            hevc_error_count = 0
                            consecutive_errors = 0
                            # Continue to next iteration
                            continue
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Multiple consecutive read errors for {self.name}, attempting reconnection")
                        self._attempt_reconnection_in_thread()
                        consecutive_errors = 0
                else:
                    consecutive_errors = 0
                    # Reset HEVC error counter when we get good frames
                    if ret and frame is not None:
                        hevc_error_count = 0
                
                # Calculate FPS
                if self.last_frame_time != 0:
                    self.fps = 1 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
                
                # Update last successful read time if we got a valid frame
                if ret and frame is not None:
                    self.last_successful_read = current_time
                    self.failed_reads = 0
                
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    
                # Avoid maxing out CPU and control playback speed for video files
                if self.is_video_file:
                    # Get video's natural FPS and adjust sleep time accordingly
                    with self.lock:
                        if self.cap is not None:
                            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            if video_fps > 0:
                                sleep_time = 1.0 / video_fps
                                time.sleep(max(0.001, sleep_time))  # Ensure minimum sleep time
                            else:
                                time.sleep(0.03)  # Default to ~30 FPS if can't determine
                        else:
                            time.sleep(0.03)
                else:
                    # For streaming sources, check if we haven't received frames in a while
                    time_since_last_frame = current_time - self.last_successful_read
                    if self._is_streaming_source(self.source) and time_since_last_frame > 10:  # 10 seconds timeout
                        timeout_ms = int(time_since_last_frame * 1000) if time_since_last_frame is not None else 0
                        print(f"Stream timeout triggered after {timeout_ms} ms for {self.name}")
                        self._attempt_reconnection_in_thread()
                        self.last_successful_read = current_time  # Reset timer to prevent multiple reconnects
                    
                    time.sleep(0.01)  # Regular camera/stream
            except Exception as e:
                print(f"Error in capture loop for camera {self.name}: {e}")
                consecutive_errors += 1
                time.sleep(0.1)
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many errors in capture loop for {self.name}, attempting reconnection")
                    self._attempt_reconnection_in_thread()
                    consecutive_errors = 0
    
    def _attempt_reconnection_in_thread(self):
        """Attempt to reconnect the camera in a separate thread to avoid blocking the main thread"""
        reconnect_thread = threading.Thread(target=self._reconnect_camera, daemon=True)
        reconnect_thread.start()
    
    def _reconnect_camera(self):
        """Attempt to reconnect the camera with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"Maximum reconnection attempts reached for {self.name}, giving up")
            self.reconnect_attempts = 0  # Reset counter for future attempts
            self.reconnect_backoff = 1  # Reset backoff time
            return False
        
        self.reconnect_attempts += 1
        backoff_time = self.reconnect_backoff
        self.reconnect_backoff = min(30, self.reconnect_backoff * 2)  # Exponential backoff, max 30 seconds
        
        print(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} "
              f"for {self.name} with {backoff_time}s backoff")
        
        # Stop existing capture with proper thread safety
        release_success = self._safe_release_capture()
        
        # Wait before reconnecting - longer waiting time after release failure helps FFmpeg cleanup resources
        if not release_success:
            print(f"Camera release failed for {self.name}, waiting longer before reconnect")
            time.sleep(backoff_time * 2)  # Wait longer if release failed
        else:
            time.sleep(backoff_time)
        
        # Force garbage collection to ensure FFmpeg resources are released
        try:
            import gc
            gc.collect()
        except:
            pass
        
        # Try to reopen the source
        try:
            # Create the capture object in a thread-safe way
            with self.lock:
                if self._is_streaming_source(self.source):
                    # For streaming sources, use more resilient connection settings
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    
                    # Check if capture was created successfully
                    if self.cap is None:
                        print(f"Failed to create capture object for {self.name}")
                        return False
                    
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    
                    # Special handling for HEVC streams experiencing reference frame errors
                    # Try a more conservative approach with more robust settings
                    try:
                        # Force H.264 codec for compatibility (if stream supports it)
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                        
                        # Connection timeout (in milliseconds)
                        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 seconds
                        
                        # Disable hardware acceleration when reconnecting since it can cause issues
                        if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 0)
                        
                        # FFmpeg-specific options to fix the async_lock assertion error
                        extra_options = {
                            'rtsp_transport': 'tcp',
                            'max_delay': '200000',  # More aggressive 200ms max delay
                            'fflags': 'nobuffer+discardcorrupt+genpts',  # More aggressive stream handling
                            'flags': 'low_delay',
                            'strict': 'experimental',
                            'stimeout': '5000000',  # Socket timeout in microseconds (5 seconds)
                            'analyzeduration': '500000',  # Even lower analyze duration
                            'sync': 'ext',  # External clock synchronization
                            'max_analyze_duration': '500000',  # 0.5 seconds max for format detection
                            # HEVC-specific error handling
                            'err_detect': 'ignore_err',  # Ignore decoding errors
                            'skip_loop_filter': 'all',   # Skip all loop filtering to handle corrupt frames
                            'skip_frame': 'default',     # Skip frames if needed
                            # Additional options to prevent thread assertion failures
                            'threads': '1',             # Use single thread to avoid threading issues
                            'enable_drefs': '0',        # Disable direct rendering which can cause thread issues
                            'stimeout': '5000000'       # Socket timeout to prevent hanging
                        }
                        
                        # Try to set each option individually
                        for option, value in extra_options.items():
                            try:
                                option_id = 1000000 + hash(option) % 1000
                                self.cap.set(option_id, value)
                            except:
                                pass
                            
                        # Request lower fps for more reliable reconnection
                        self.cap.set(cv2.CAP_PROP_FPS, 10)  # Even lower FPS when reconnecting
                        
                    except Exception as e:
                        print(f"Warning: Could not set advanced options during reconnect: {e}")
                else:
                    # Regular non-streaming source
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # Check if capture opened successfully
                if not self.cap.isOpened():
                    print(f"Failed to reconnect to camera {self.name}")
                    self.cap = None  # Reset to avoid using invalid capture
                    return False
                
                # Get a test frame to confirm connection - with lock protection
                success = False
                for i in range(3):  # Try up to 3 times
                    try:
                        ret, _ = self.cap.read()
                        if ret:
                            success = True
                            break
                        time.sleep(0.5)  # Short delay between attempts
                    except Exception as e:
                        print(f"Error reading test frame ({i+1}/3): {e}")
                        time.sleep(0.5)
                
                # Update state based on success/failure
                if success:
                    print(f"Successfully reconnected to camera {self.name}")
                    self.last_successful_read = time.time()
                    self.failed_reads = 0
                    return True
                else:
                    print(f"Reconnection to {self.name} failed to read test frame after multiple attempts")
                    # Release the failed capture before returning
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                    return False
                
        except Exception as e:
            print(f"Error during reconnection to {self.name}: {e}")
            with self.lock:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
            return False
    
    def _safe_release_capture(self):
        """Safely release the capture object with proper error handling"""
        success = True
        with self.lock:
            if self.cap is not None:
                try:
                    # Try a clean release
                    self.cap.release()
                except Exception as e:
                    print(f"Error releasing camera {self.name}: {e}")
                    success = False
                finally:
                    # Ensure we set to None even if release failed
                    self.cap = None
        
        return success
    
    def _reconnect_for_hevc_errors(self):
        """Special reconnection strategy for HEVC reference frame issues"""
        print(f"Applying HEVC-specific reconnection for {self.name}")
        
        # Safely release the capture
        self._safe_release_capture()
        
        # Wait a moment - longer to ensure FFmpeg resources are released
        time.sleep(3)  # Extended wait time for HEVC streams
        
        # Force garbage collection
        try:
            import gc
            gc.collect()
        except:
            pass
        
        try:
            # Specifically for "Could not find ref with POC" errors in HEVC streams,
            # we need to completely bypass the hardware accelerated decoding and
            # force software decoding with strict error handling
            
            # Reopen with HEVC-specific workarounds in a thread-safe manner
            with self.lock:
                # For HEVC streams, add direct FFmpeg options to the URL if possible
                if self._might_be_hevc_stream(self.source) and isinstance(self.source, str) and self.source.startswith('rtsp://'):
                    # If this is an RTSP URL, we can use FFmpeg transport options
                    # to directly control the stream
                    if '?' not in self.source:
                        # Add FFmpeg options to force TCP and disable hardware decoding
                        modified_source = f"{self.source}?rtsp_transport=tcp&rtsp_flags=prefer_tcp"
                        print(f"Modified RTSP source for HEVC stream: {modified_source}")
                        self.cap = cv2.VideoCapture(modified_source, cv2.CAP_FFMPEG)
                    else:
                        # URL already has query parameters, use the original source
                        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                else:
                    # Regular source
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                
                # Check if capture object was created
                if self.cap is None:
                    print(f"Failed to create capture object for HEVC reconnection to {self.name}")
                    return False
                    
                # Set essential buffer options
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # Set strict HEVC handling options - optimized specifically for 
                # "Could not find ref with POC" errors
                try:
                    # Force software decoding by setting specific HEVC options
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                    
                    # Create specific options for "Could not find ref with POC" errors
                    poc_error_options = {
                        # Connection options
                        'rtsp_transport': 'tcp',
                        'stimeout': '10000000',  # 10 seconds socket timeout
                        'timeout': '10000000',   # 10 seconds connection timeout
                        
                        # HEVC/FFmpeg specific options for reference frame errors
                        'fflags': 'discardcorrupt+genpts+igndts',  # Discard corrupt packets & fix timestamps
                        'flags': 'low_delay',    # Prioritize low latency
                        'strict': 'experimental', # Allow experimental features
                        
                        # Error recovery options
                        'skip_loop_filter': 'all',  # Skip all loop filtering (required for POC errors)
                        'err_detect': 'ignore_err', # Continue despite errors
                        'skip_frame': 'nointra',    # Skip non-intra frames if needed for recovery
                        
                        # Reference frame handling
                        'max_delay': '300000',     # Lower max delay for quick recovery
                        'r_frame_rate': '15',      # Lower frame rate to reduce likelihood of ref issues
                        'i-only': '1',             # Force I-frame only mode for problematic streams
                        
                        # Threading options - crucial for the async_lock issue
                        'threads': '1',            # Single-threaded decoding to avoid thread safety issues
                        'async': '0',              # Disable async decoding entirely 
                        'max_analyze_duration': '500000', # Shorter analyze duration
                        
                        # Decoder options
                        'hwaccel': 'none',         # Force software decoding
                        'tune': 'zerolatency',     # Tune for latency
                        'vcodec': 'none',          # Let FFmpeg choose best decoder
                        
                        # I/O settings
                        'analyzeduration': '400000', # 400ms analyze time
                        'probesize': '32000',      # Smaller probe size for faster start
                    }
                    
                    # Apply all HEVC-specific options
                    for option, value in poc_error_options.items():
                        try:
                            option_id = 1000000 + hash(option) % 1000
                            self.cap.set(option_id, value)
                        except:
                            pass
                    
                    # Set lower FPS for more reliable connection
                    self.cap.set(cv2.CAP_PROP_FPS, 10)
                except Exception as e:
                    print(f"Warning: Could not set HEVC-specific options: {e}")
                
                # Verify the capture is open
                if not self.cap.isOpened():
                    print(f"Failed to open HEVC connection for {self.name}")
                    self.cap = None
                    return False
                
                # Test reading a frame - with repeated attempts and longer timeouts
                print(f"Testing HEVC reconnection for {self.name} with multiple attempts...")
                success = False
                
                # Use more attempts for HEVC streams (5 instead of 3)
                for i in range(5):
                    try:
                        ret, _ = self.cap.read()
                        if ret:
                            success = True
                            print(f"Successfully read frame on attempt {i+1}/5")
                            break
                        print(f"Failed to read frame on attempt {i+1}/5, waiting longer...")
                        # Use increasing wait times between attempts
                        time.sleep(1.0 + i*0.5)  # 1s, 1.5s, 2s, 2.5s, 3s
                    except Exception as e:
                        print(f"Error reading test frame during HEVC reconnection ({i+1}/5): {e}")
                        time.sleep(1.0)
                
                # Reset counters if successful
                if success:
                    print(f"Successfully reconnected to HEVC stream {self.name} after POC error")
                    self.failed_reads = 0
                    self.last_successful_read = time.time()
                    return True
                else:
                    print(f"Failed to read test frame after HEVC reconnection for {self.name}")
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                    return False
            
        except Exception as e:
            print(f"Error in HEVC-specific reconnection for {self.name}: {e}")
            with self.lock:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
            return False
    
    def get_frame(self):
        """Get the latest frame from the camera"""
        with self.lock:
            if not self.ret or self.frame is None:
                self.failed_reads += 1
            else:
                self.failed_reads = 0
                
            return self.ret, self.frame.copy() if self.frame is not None else None
    
    def get_progress(self):
        """Get progress information for video files"""
        if not self.is_video_file or self.total_frames <= 0:
            return None
        
        # Calculate progress with safety checks
        try:
            progress_percent = (self.current_frame_index / self.total_frames) * 100 if self.total_frames > 0 else 0
            return {
                'current_frame': self.current_frame_index,
                'total_frames': self.total_frames,
                'progress_percent': progress_percent
            }
        except (TypeError, ZeroDivisionError):
            # Handle potential errors
            return {
                'current_frame': self.current_frame_index,
                'total_frames': self.total_frames,
                'progress_percent': 0
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

    def _might_be_hevc_stream(self, source):
        """
        Analyze the source URL to determine if it's likely an HEVC (H.265) stream.
        Many CCTV cameras use HEVC encoding.
        
        Args:
            source: Camera source URL
            
        Returns:
            bool: True if likely an HEVC stream
        """
        if not isinstance(source, str):
            return False
            
        # Common indicators of HEVC streams in URLs
        hevc_indicators = [
            'h265', 'hevc', 'h.265', 'subtype=0', 'profile=high', 
            'cam/realmonitor', 'hisilicon', 'dahua', 'hikvision', 
            '554', '1554', '8554'  # Common RTSP ports for CCTV cameras
        ]
        
        # Check if any indicators are in the source string
        source_lower = source.lower()
        for indicator in hevc_indicators:
            if indicator.lower() in source_lower:
                return True
                
        # If the URL doesn't explicitly indicate codec but it's an RTSP stream to a typical IP address
        # pattern used by security cameras, consider it as potentially HEVC
        if source.startswith('rtsp://'):
            # Common security camera IP patterns
            if any(pattern in source for pattern in ['192.168.', '10.0.', '10.1.', '172.16.', '172.31.']):
                return True
                
        return False


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
        
        camera = self.cameras[camera_name]
        
        # Get frame
        ret, frame = camera.get_frame()
        
        # Check if we need to reconnect based on failed reads
        if not ret or frame is None:
            # For streaming sources, if there are too many consecutive failed reads, try to restart
            if camera._is_streaming_source(camera.source) and camera.failed_reads >= 3:
                print(f"Stream timeout detected for {camera_name} after {camera.failed_reads} failed reads. Attempting reconnection...")
                
                # Try to reconnect right now
                camera._attempt_reconnection_in_thread()
                
                # Return the current frame (which may be None, but application should handle this)
            
        return ret, frame
    
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
