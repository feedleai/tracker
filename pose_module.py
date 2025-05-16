import numpy as np
import cv2
import time
import os

class PoseModule:
    """
    Module for human pose estimation to enhance gait recognition and tracking
    """
    def __init__(self, config=None):
        """
        Initialize the pose estimation module
        
        Args:
            config (dict): Configuration parameters for pose estimation
        """
        self.config = config or {}
        
        # Load pose estimation model
        self.model_path = self.config.get('model_path', 'models/pose')
        self.conf_threshold = self.config.get('confidence_threshold', 0.5)
        self.using_gpu = self.config.get('use_gpu', True)
        
        # Initialize pose model
        self._initialize_model()
        
        # Keypoint indices for different body parts
        self.keypoint_indices = {
            'nose': 0,
            'neck': 1,
            'right_shoulder': 2,
            'right_elbow': 3,
            'right_wrist': 4,
            'left_shoulder': 5,
            'left_elbow': 6,
            'left_wrist': 7,
            'right_hip': 8,
            'right_knee': 9,
            'right_ankle': 10,
            'left_hip': 11,
            'left_knee': 12,
            'left_ankle': 13,
            'right_eye': 14,
            'left_eye': 15,
            'right_ear': 16,
            'left_ear': 17,
        }
        
        # Track pose data history for each person
        self.pose_history = {}  # track_id -> list of pose data
        self.max_history = self.config.get('max_history', 30)  # frames
        
    def _initialize_model(self):
        """Initialize the OpenCV-based pose estimation model"""
        try:
            # Check if model paths exist
            openpose_prototxt = f"{self.model_path}/pose_deploy_linevec.prototxt"
            openpose_model = f"{self.model_path}/pose_iter_440000.caffemodel"
            movenet_model = f"{self.model_path}/movenet_lightning.pb"
            
            use_openpose = os.path.exists(openpose_prototxt) and os.path.exists(openpose_model)
            use_movenet = os.path.exists(movenet_model)
            
            if not use_openpose and not use_movenet:
                print(f"No pose estimation models found in {self.model_path}")
                print("Download the models or update the model_path in the configuration")
                self.net = None
                self.model_type = None
                return
            
            # Try to load OpenPose model first if files exist
            if use_openpose:
                try:
                    self.net = cv2.dnn.readNetFromCaffe(openpose_prototxt, openpose_model)
                    self.model_type = "openpose"
                    self.num_points = 18
                    print(f"Loaded OpenPose model for pose estimation from {openpose_prototxt}")
                except Exception as e:
                    print(f"Failed to load OpenPose model: {e}")
                    use_openpose = False
            
            # Fall back to MoveNet model if OpenPose failed or files don't exist
            if not use_openpose and use_movenet:
                try:
                    self.net = cv2.dnn.readNetFromTensorflow(movenet_model)
                    self.model_type = "movenet"
                    self.num_points = 17
                    print(f"Loaded MoveNet model for pose estimation from {movenet_model}")
                except Exception as e:
                    print(f"Failed to load MoveNet model: {e}")
                    self.net = None
                    self.model_type = None
                    return
            
            # Configure model and backend
            if self.net is not None and self.using_gpu:
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("Using GPU for pose estimation")
                except Exception as e:
                    print(f"GPU acceleration failed for pose estimation: {e}")
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print("Falling back to CPU for pose estimation")
            elif self.net is not None:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU for pose estimation")
                
        except Exception as e:
            print(f"Error initializing pose model: {e}")
            print("Continuing without pose estimation")
            self.net = None
            self.model_type = None
    
    def detect_poses(self, frame, tracked_persons):
        """
        Detect human poses in the frame for each tracked person
        
        Args:
            frame (numpy.ndarray): Input video frame
            tracked_persons (dict): Dictionary of tracked persons
            
        Returns:
            dict: Updated dictionary of tracked persons with pose data
        """
        if self.net is None:
            return tracked_persons
        
        frame_height, frame_width = frame.shape[:2]
        
        # Process each tracked person
        for track_id, person in tracked_persons.items():
            if 'bbox' not in person:
                continue
                
            bbox = person['bbox']
            
            # Get person crop with margin
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Add some margin around the bounding box
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(frame_width, x2 + margin_x)
            y2 = min(frame_height, y2 + margin_y)
            
            # Skip if bounding box is invalid
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Extract person crop
            person_crop = frame[y1:y2, x1:x2]
            
            # Skip empty crops
            if person_crop.size == 0:
                continue
            
            # Detect pose
            keypoints = self._detect_pose_in_crop(person_crop)
            
            if keypoints is not None:
                # Convert keypoint coordinates to original frame coordinates
                for i in range(len(keypoints)):
                    if keypoints[i] is not None:
                        # Convert relative coordinates within crop to absolute coordinates in frame
                        keypoints[i][0] = keypoints[i][0] * (x2 - x1) + x1
                        keypoints[i][1] = keypoints[i][1] * (y2 - y1) + y1
                
                # Add pose keypoints to person data
                person['keypoints'] = keypoints
                
                # Calculate joint angles and leg stride features for gait analysis
                person['joint_angles'] = self._calculate_joint_angles(keypoints)
                person['leg_stride'] = self._calculate_leg_stride(keypoints)
                
                # Update pose history
                if track_id not in self.pose_history:
                    self.pose_history[track_id] = []
                
                # Store relevant pose features for gait analysis
                pose_features = {
                    'keypoints': keypoints,
                    'joint_angles': person['joint_angles'],
                    'leg_stride': person['leg_stride'],
                    'timestamp': time.time()
                }
                
                self.pose_history[track_id].append(pose_features)
                
                # Keep only the most recent frames
                if len(self.pose_history[track_id]) > self.max_history:
                    self.pose_history[track_id] = self.pose_history[track_id][-self.max_history:]
        
        return tracked_persons
    
    def _detect_pose_in_crop(self, crop):
        """
        Detect pose keypoints in a cropped person image
        
        Args:
            crop (numpy.ndarray): Cropped person image
            
        Returns:
            list: List of keypoints (x, y, confidence) or None if detection failed
        """
        if self.net is None:
            return None
            
        try:
            crop_height, crop_width = crop.shape[:2]
            
            if self.model_type == "openpose":
                # OpenPose model processing
                blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (368, 368), (127.5, 127.5, 127.5), swapRB=True)
                self.net.setInput(blob)
                output = self.net.forward()
                
                # Process the heatmaps to get keypoints
                keypoints = []
                for i in range(self.num_points):
                    # Get confidence map for this keypoint
                    probMap = output[0, i, :, :]
                    probMap = cv2.resize(probMap, (crop_width, crop_height))
                    
                    # Find global maxima of the probability map
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                    
                    # Add keypoint if confidence is high enough
                    if prob > self.conf_threshold:
                        # Normalize coordinates
                        x = point[0] / crop_width
                        y = point[1] / crop_height
                        keypoints.append([x, y, prob])
                    else:
                        keypoints.append(None)
                        
            elif self.model_type == "movenet":
                # MoveNet model processing
                blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (192, 192), (127.5, 127.5, 127.5), swapRB=True)
                self.net.setInput(blob)
                output = self.net.forward()
                
                # Process output to get keypoints
                keypoints = []
                for i in range(self.num_points):
                    y = output[0, i, 0]
                    x = output[0, i, 1]
                    confidence = output[0, i, 2]
                    
                    # Add keypoint if confidence is high enough
                    if confidence > self.conf_threshold:
                        keypoints.append([x, y, confidence])
                    else:
                        keypoints.append(None)
            else:
                return None
                
            return keypoints
            
        except Exception as e:
            print(f"Error detecting pose: {e}")
            return None
    
    def _calculate_joint_angles(self, keypoints):
        """
        Calculate joint angles for gait analysis
        
        Args:
            keypoints (list): List of keypoints
            
        Returns:
            dict: Dictionary of joint angles
        """
        joint_angles = {}
        
        # Helper function to calculate angle between three points
        def calculate_angle(a, b, c):
            if a is None or b is None or c is None:
                return None
            
            # Convert to numpy arrays for easier calculation
            a = np.array([a[0], a[1]])
            b = np.array([b[0], b[1]])
            c = np.array([c[0], c[1]])
            
            # Create vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure valid range
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        
        # Calculate hip angles
        if all(keypoints[i] is not None for i in [self.keypoint_indices[k] for k in ['right_hip', 'right_knee', 'right_ankle']]):
            joint_angles['right_knee'] = calculate_angle(
                keypoints[self.keypoint_indices['right_hip']],
                keypoints[self.keypoint_indices['right_knee']],
                keypoints[self.keypoint_indices['right_ankle']]
            )
        
        if all(keypoints[i] is not None for i in [self.keypoint_indices[k] for k in ['left_hip', 'left_knee', 'left_ankle']]):
            joint_angles['left_knee'] = calculate_angle(
                keypoints[self.keypoint_indices['left_hip']],
                keypoints[self.keypoint_indices['left_knee']],
                keypoints[self.keypoint_indices['left_ankle']]
            )
        
        # Calculate hip-knee-ankle system (lower body posture)
        if all(keypoints[i] is not None for i in [self.keypoint_indices[k] for k in ['left_hip', 'right_hip', 'left_knee', 'right_knee']]):
            hip_center = [(keypoints[self.keypoint_indices['left_hip']][0] + keypoints[self.keypoint_indices['right_hip']][0]) / 2,
                          (keypoints[self.keypoint_indices['left_hip']][1] + keypoints[self.keypoint_indices['right_hip']][1]) / 2]
                          
            knee_center = [(keypoints[self.keypoint_indices['left_knee']][0] + keypoints[self.keypoint_indices['right_knee']][0]) / 2,
                           (keypoints[self.keypoint_indices['left_knee']][1] + keypoints[self.keypoint_indices['right_knee']][1]) / 2]
                           
            joint_angles['hip_knee_vertical'] = calculate_angle(
                [hip_center[0], 0],  # Point above hip
                hip_center,
                knee_center
            )
        
        return joint_angles
    
    def _calculate_leg_stride(self, keypoints):
        """
        Calculate leg stride parameters for gait analysis
        
        Args:
            keypoints (list): List of keypoints
            
        Returns:
            dict: Dictionary of stride parameters
        """
        stride_params = {}
        
        # Check if we have ankle keypoints
        left_ankle_idx = self.keypoint_indices.get('left_ankle')
        right_ankle_idx = self.keypoint_indices.get('right_ankle')
        
        if (left_ankle_idx is not None and right_ankle_idx is not None and
            keypoints[left_ankle_idx] is not None and keypoints[right_ankle_idx] is not None):
            
            # Calculate horizontal distance between ankles (stride length)
            left_x, left_y = keypoints[left_ankle_idx][0], keypoints[left_ankle_idx][1]
            right_x, right_y = keypoints[right_ankle_idx][0], keypoints[right_ankle_idx][1]
            
            stride_length = abs(left_x - right_x)
            stride_params['stride_length'] = stride_length
            
            # Calculate vertical alignment of ankles
            stride_params['ankle_y_diff'] = abs(left_y - right_y)
            
            # Calculate ankle positions relative to body
            hip_center_idx = None
            if (self.keypoint_indices.get('left_hip') is not None and 
                self.keypoint_indices.get('right_hip') is not None and
                keypoints[self.keypoint_indices['left_hip']] is not None and 
                keypoints[self.keypoint_indices['right_hip']] is not None):
                
                hip_center_x = (keypoints[self.keypoint_indices['left_hip']][0] + 
                              keypoints[self.keypoint_indices['right_hip']][0]) / 2
                              
                # Calculate ankle forward/backward positions relative to hip center
                stride_params['left_ankle_offset'] = left_x - hip_center_x
                stride_params['right_ankle_offset'] = right_x - hip_center_x
                
        return stride_params
    
    def extract_pose_features(self, track_id):
        """
        Extract pose-based features for gait recognition from historical data
        
        Args:
            track_id (int): Tracking ID of the person
            
        Returns:
            numpy.ndarray or None: Extracted pose features for gait analysis
        """
        if track_id not in self.pose_history or len(self.pose_history[track_id]) < 5:
            return None
            
        # Get pose history for this track
        pose_history = self.pose_history[track_id]
        
        # Extract features from pose history
        features = []
        
        # 1. Knee angle patterns and variability
        left_knee_angles = [p['joint_angles'].get('left_knee', 0) for p in pose_history 
                           if p['joint_angles'].get('left_knee') is not None]
        right_knee_angles = [p['joint_angles'].get('right_knee', 0) for p in pose_history 
                            if p['joint_angles'].get('right_knee') is not None]
        
        if left_knee_angles and right_knee_angles:
            # Calculate statistics of joint angles
            mean_left_knee = np.mean(left_knee_angles)
            std_left_knee = np.std(left_knee_angles)
            mean_right_knee = np.mean(right_knee_angles)
            std_right_knee = np.std(right_knee_angles)
            
            # Calculate angle range (max flexion/extension)
            range_left_knee = np.max(left_knee_angles) - np.min(left_knee_angles) if left_knee_angles else 0
            range_right_knee = np.max(right_knee_angles) - np.min(right_knee_angles) if right_knee_angles else 0
            
            features.extend([mean_left_knee, std_left_knee, range_left_knee,
                           mean_right_knee, std_right_knee, range_right_knee])
        else:
            # Placeholder values
            features.extend([0, 0, 0, 0, 0, 0])
        
        # 2. Stride parameters
        stride_lengths = [p['leg_stride'].get('stride_length', 0) for p in pose_history 
                         if p['leg_stride'].get('stride_length') is not None]
        
        ankle_y_diffs = [p['leg_stride'].get('ankle_y_diff', 0) for p in pose_history 
                        if p['leg_stride'].get('ankle_y_diff') is not None]
        
        left_offsets = [p['leg_stride'].get('left_ankle_offset', 0) for p in pose_history 
                       if p['leg_stride'].get('left_ankle_offset') is not None]
        
        right_offsets = [p['leg_stride'].get('right_ankle_offset', 0) for p in pose_history 
                        if p['leg_stride'].get('right_ankle_offset') is not None]
        
        if stride_lengths:
            mean_stride = np.mean(stride_lengths)
            std_stride = np.std(stride_lengths)
            max_stride = np.max(stride_lengths)
            
            mean_ankle_y_diff = np.mean(ankle_y_diffs) if ankle_y_diffs else 0
            
            # Stride asymmetry (difference between left and right ankle offsets)
            left_offset_mean = np.mean(left_offsets) if left_offsets else 0
            right_offset_mean = np.mean(right_offsets) if right_offsets else 0
            offset_asymmetry = abs(left_offset_mean - right_offset_mean)
            
            features.extend([mean_stride, std_stride, max_stride, mean_ankle_y_diff, offset_asymmetry])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 3. Calculate temporal patterns (cycle frequency)
        # For simplicity, use knee angle peaks as a proxy for steps
        if len(left_knee_angles) > 6:
            # Find peaks in knee angle patterns (simplified)
            from scipy.signal import find_peaks
            
            # Try to find peaks in left knee angles
            try:
                peaks, _ = find_peaks(left_knee_angles, distance=3)
                left_cycle_freq = len(peaks) / len(left_knee_angles) if peaks.size > 0 else 0
            except:
                left_cycle_freq = 0
                
            # Try to find peaks in right knee angles
            try:
                peaks, _ = find_peaks(right_knee_angles, distance=3)
                right_cycle_freq = len(peaks) / len(right_knee_angles) if peaks.size > 0 else 0
            except:
                right_cycle_freq = 0
                
            features.extend([left_cycle_freq, right_cycle_freq])
        else:
            features.extend([0, 0])
        
        # Convert to numpy array and ensure all values are finite
        features_array = np.array(features, dtype=np.float32)
        features_array = np.nan_to_num(features_array)  # Replace NaN with 0
        
        return features_array
    
    def draw_poses(self, frame, tracked_persons):
        """
        Draw detected poses on the output frame
        
        Args:
            frame (numpy.ndarray): Input frame to draw on
            tracked_persons (dict): Dictionary of tracked persons with pose data
            
        Returns:
            numpy.ndarray: Frame with poses drawn
        """
        result_frame = frame.copy()
        
        # Define pairs of keypoints for drawing skeleton lines
        pairs = [
            ('nose', 'neck'),
            ('neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('neck', 'left_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('neck', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
            ('neck', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('nose', 'right_eye'), ('right_eye', 'right_ear'),
            ('nose', 'left_eye'), ('left_eye', 'left_ear')
        ]
        
        for track_id, person in tracked_persons.items():
            try:
                if 'keypoints' not in person or person['keypoints'] is None:
                    continue
                    
                keypoints = person['keypoints']
                if not keypoints:
                    continue
                
                # Draw skeleton lines
                for pair in pairs:
                    try:
                        idx1 = self.keypoint_indices.get(pair[0])
                        idx2 = self.keypoint_indices.get(pair[1])
                        
                        if (idx1 is not None and idx2 is not None and 
                            idx1 < len(keypoints) and idx2 < len(keypoints) and 
                            keypoints[idx1] is not None and keypoints[idx2] is not None):
                            
                            pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                            pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                            
                            # Verify points are in image bounds
                            h, w = result_frame.shape[:2]
                            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                cv2.line(result_frame, pt1, pt2, (0, 255, 255), 2)
                    except (ValueError, TypeError, IndexError) as e:
                        # Skip this pair if there's an error
                        continue
                
                # Draw keypoints
                for i, keypoint in enumerate(keypoints):
                    try:
                        if keypoint is not None and len(keypoint) >= 2:
                            x, y = int(keypoint[0]), int(keypoint[1])
                            
                            # Verify point is in image bounds
                            h, w = result_frame.shape[:2]
                            if 0 <= x < w and 0 <= y < h:
                                cv2.circle(result_frame, (x, y), 4, (0, 255, 0), -1)
                    except (ValueError, TypeError, IndexError) as e:
                        # Skip this keypoint if there's an error
                        continue
                
                # Add gait-related information if available
                try:
                    if ('joint_angles' in person and 'leg_stride' in person and
                        person['joint_angles'] is not None and person['leg_stride'] is not None):
                        
                        stride_length = person['leg_stride'].get('stride_length', 0)
                        left_knee = person['joint_angles'].get('left_knee', 0)
                        right_knee = person['joint_angles'].get('right_knee', 0)
                        
                        # Get bounding box and draw info
                        if 'bbox' in person and person['bbox'] is not None:
                            bbox = person['bbox']
                            if len(bbox) == 4:
                                x1, y1 = int(bbox[0]), int(bbox[1])
                                
                                info_text = f"Stride: {stride_length:.1f}"
                                cv2.putText(result_frame, info_text, (x1, y1 - 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                knee_text = f"L: {left_knee:.1f}° R: {right_knee:.1f}°"
                                cv2.putText(result_frame, knee_text, (x1, y1 - 25), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    # Skip gait info if there's an error
                    continue
            except Exception as e:
                print(f"Error drawing pose for track {track_id}: {e}")
        
        return result_frame 