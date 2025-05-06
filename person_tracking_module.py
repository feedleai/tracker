import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


class PersonTrackingModule:
    def __init__(self, config):
        """
        Initialize the person detection and tracking module using YOLOv8 and BoTSORT.
        
        Args:
            config (dict): Configuration parameters for person tracking
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.7)
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        
        # Configure GPU/CPU usage
        self.use_gpu = config.get('use_gpu', True)
        self.device = 'cpu'
        
        # Check if GPU is available
        try:
            import torch
            if self.use_gpu and torch.cuda.is_available():
                self.device = 'cuda:0'
                print("Using GPU acceleration for person tracking")
            else:
                print("GPU acceleration not available for person tracking, using CPU")
        except ImportError:
            print("PyTorch not available, using CPU for person tracking")
        
        # Load YOLO model - using YOLOv8n by default
        model_path = config.get('model_path', 'yolov8n.pt')
        
        # Load YOLO model with tracking enabled using BoTSORT
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Set device for inference
        self.model.to(self.device)
        
        # Class ID for person in COCO dataset (used by YOLO) is 0
        self.person_class_id = 0
        
        # Store tracked persons
        self.tracked_persons = {}
        
        # Map between tracker IDs and face IDs
        self.tracker_to_face_map = {}
        
        # Map between tracker IDs and person IDs
        self.tracker_to_person_map = {}
        
        # Store appearance features for each person to help with reidentification
        self.person_appearance_features = {}
        
        # Cache of recently disappeared tracks to help with reidentification
        self.disappeared_tracks = {}
        self.max_disappearance_frames = 120  # ~4 seconds at 30 FPS - increased for better persistence
        
        # Track velocity of persons for motion prediction
        self.person_velocities = {}
        
        # Kalman filter parameters for better tracking
        self.use_kalman = config.get('use_kalman', True)
        self.kalman_filters = {}  # Mapping of track_id to KalmanFilter objects
        self.kalman_states = {}   # Mapping of track_id to last Kalman state
        
        # More sophisticated appearance feature extraction
        self.use_deep_features = False
        try:
            # Check if we can use OpenCV's DNN module for deeper features
            cv2.dnn.readNet
            self.use_deep_features = config.get('use_deep_features', True)
            if self.use_deep_features:
                print("Using deep appearance features for better reidentification")
        except (AttributeError, ImportError):
            print("OpenCV DNN module not available, using basic appearance features")
        
        # Tracking consistency metrics
        self.track_history = {}
        self.max_track_history = 30  # Store last 30 positions for each track
        
        # Motion model for predicting person movements
        self.motion_prediction_enabled = config.get('motion_prediction', True)
        
        # Occlusion handling - actively track through occlusions
        self.occlusion_threshold = config.get('occlusion_threshold', 0.4)
        self.actively_track_occlusions = config.get('track_occlusions', True)
        
        # Frame dimensions for motion prediction
        self.frame_width = 1920  # Default, will be updated with actual frame dimensions
        self.frame_height = 1080
        
        # Frame counter
        self.frame_count = 0
        
        # Store last frame for optical flow calculations
        self.last_frame = None
        self.last_frame_gray = None
        
    def detect_and_track(self, frame):
        """
        Detect and track persons in the given frame using YOLOv8 and BoTSORT.
        
        Args:
            frame (numpy.ndarray): Input image frame
            
        Returns:
            dict: Dictionary of tracked persons with tracker IDs as keys
        """
        # Initialize empty result in case of early return
        tracked_persons = {}
        
        # Input validation
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("Warning: Invalid frame passed to detect_and_track")
            return tracked_persons
        
        try:
            # Update frame dimensions
            self.frame_height, self.frame_width = frame.shape[:2]
            
            # Increment frame counter
            self.frame_count += 1
            
            # Convert frame to grayscale for optical flow
            try:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error converting frame to grayscale: {e}")
                frame_gray = None
            
            # Calculate optical flow if we have previous frame
            flow_vectors = {}
            if self.motion_prediction_enabled and self.last_frame_gray is not None and frame_gray is not None:
                # Calculate optical flow for better motion prediction
                try:
                    # Using Lucas-Kanade method for sparse optical flow
                    for track_id, person in self.tracked_persons.items():
                        if 'bbox' not in person or person['bbox'] is None:
                            continue
                            
                        bbox = person['bbox']
                        if not isinstance(bbox, (list, np.ndarray)) or len(bbox) != 4:
                            continue
                            
                        # Define keypoints in the bounding box
                        try:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            # Create a grid of points within the bounding box
                            points = []
                            step = max(5, min((x2-x1)//4, (y2-y1)//4))  # Grid spacing based on bbox size
                            
                            # Add more points for larger bounding boxes
                            for x in range(x1, x2, step):
                                for y in range(y1, y2, step):
                                    if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                                        points.append([float(x), float(y)])
                            
                            # Add center point and corners for better coverage
                            if 0 <= center_x < self.frame_width and 0 <= center_y < self.frame_height:
                                points.append([float(center_x), float(center_y)])
                                
                            # Add corners only if they're within frame bounds
                            if 0 <= x1 < self.frame_width and 0 <= y1 < self.frame_height:
                                points.append([float(x1), float(y1)])
                            if 0 <= x2 < self.frame_width and 0 <= y1 < self.frame_height:    
                                points.append([float(x2), float(y1)])
                            if 0 <= x1 < self.frame_width and 0 <= y2 < self.frame_height:
                                points.append([float(x1), float(y2)])
                            if 0 <= x2 < self.frame_width and 0 <= y2 < self.frame_height:
                                points.append([float(x2), float(y2)])
                            
                            if points:
                                points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                                
                                # Validate shapes match requirements for calcOpticalFlowPyrLK
                                if points.shape[0] > 0 and self.last_frame_gray.shape == frame_gray.shape:
                                    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                                        self.last_frame_gray, 
                                        frame_gray, 
                                        points, 
                                        None, 
                                        winSize=(15, 15), 
                                        maxLevel=2,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                                    )
                                    
                                    # Filter valid points
                                    good_new = []
                                    good_old = []
                                    for i, (new, old, stat) in enumerate(zip(new_points, points, status)):
                                        if stat == 1:
                                            good_new.append(new.ravel())
                                            good_old.append(old.ravel())
                                    
                                    good_points = np.array(good_old) if good_old else None
                                    good_new_points = np.array(good_new) if good_new else None
                                    
                                    if good_points is not None and good_new_points is not None and len(good_points) > 0:
                                        # Calculate average motion vector
                                        dx = np.mean(good_new_points[:, 0] - good_points[:, 0])
                                        dy = np.mean(good_new_points[:, 1] - good_points[:, 1])
                                        
                                        # Remove outliers for more robust estimation
                                        if len(good_points) > 4:
                                            # Calculate individual motion vectors
                                            motion_vectors = good_new_points - good_points
                                            
                                            # Calculate mean and standard deviation of motion
                                            mean_dx = np.mean(motion_vectors[:, 0])
                                            mean_dy = np.mean(motion_vectors[:, 1])
                                            std_dx = np.std(motion_vectors[:, 0])
                                            std_dy = np.std(motion_vectors[:, 1])
                                            
                                            # Filter points within 2 standard deviations
                                            inliers = np.logical_and(
                                                np.abs(motion_vectors[:, 0] - mean_dx) < 2 * std_dx,
                                                np.abs(motion_vectors[:, 1] - mean_dy) < 2 * std_dy
                                            )
                                            
                                            if np.sum(inliers) > 0:
                                                # Recalculate mean with inliers only
                                                dx = np.mean(motion_vectors[inliers, 0])
                                                dy = np.mean(motion_vectors[inliers, 1])
                                        
                                        # Store flow vector
                                        flow_vectors[track_id] = (dx, dy)
                        except Exception as e:
                            print(f"Error processing optical flow points for track {track_id}: {e}")
                except Exception as e:
                    print(f"Error calculating optical flow: {e}")
            
            # Run the model with tracking enabled
            try:
                results = self.model.track(
                    frame, 
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    persist=True,  # Remember tracks between frames
                    tracker="botsort.yaml",  # Use BoTSORT tracker
                    device=self.device  # Specify device for inference
                )
            except Exception as e:
                print(f"Error running YOLO model: {e}")
                # Create empty results to avoid crashes
                results = None
            
            # First, predict new positions for previously tracked persons
            predicted_boxes = {}
            if self.motion_prediction_enabled:
                for track_id, person in self.tracked_persons.items():
                    try:
                        if 'bbox' not in person or person['bbox'] is None:
                            continue
                            
                        # Get predicted motion from optical flow
                        dx, dy = 0, 0
                        if track_id in flow_vectors:
                            dx, dy = flow_vectors[track_id]
                        
                        # Get velocity from history
                        vx, vy = 0, 0
                        if track_id in self.person_velocities:
                            vx, vy = self.person_velocities[track_id]
                            
                        # Combine flow and velocity for prediction with better outlier rejection
                        flow_magnitude = np.sqrt(dx**2 + dy**2)
                        velocity_magnitude = np.sqrt(vx**2 + vy**2)
                        
                        # Use flow only if it's not too large (reject erratic motions)
                        if flow_magnitude < 30 and abs(flow_magnitude - velocity_magnitude) < 20:
                            # Weighted average of flow and velocity (favor flow for fast adaptation)
                            pred_dx = 0.7 * dx + 0.3 * vx
                            pred_dy = 0.7 * dy + 0.3 * vy
                        else:
                            # Flow is erratic, use velocity only
                            pred_dx = vx
                            pred_dy = vy
                        
                        # Predict new bounding box
                        bbox = person['bbox'].copy()
                        predicted_box = [
                            bbox[0] + pred_dx,
                            bbox[1] + pred_dy,
                            bbox[2] + pred_dx,
                            bbox[3] + pred_dy
                        ]
                        
                        # Ensure predicted box is within frame
                        predicted_box[0] = max(0, min(predicted_box[0], self.frame_width-1))
                        predicted_box[1] = max(0, min(predicted_box[1], self.frame_height-1))
                        predicted_box[2] = max(0, min(predicted_box[2], self.frame_width-1))
                        predicted_box[3] = max(0, min(predicted_box[3], self.frame_height-1))
                        
                        predicted_boxes[track_id] = predicted_box
                    except Exception as e:
                        print(f"Error predicting box for track {track_id}: {e}")
            
            # Process YOLO detections
            if results and len(results) > 0:
                try:
                    # Extract tracking results
                    detections = results[0]
                    
                    if detections.boxes is not None and len(detections.boxes) > 0:
                        boxes = detections.boxes
                        
                        for i, box in enumerate(boxes):
                            try:
                                # Check if this is a person class detection
                                cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                                if cls != self.person_class_id:
                                    continue
                                    
                                # Get tracking ID if available
                                if box.id is not None:
                                    track_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)
                                else:
                                    continue  # Skip detections without tracking IDs
                                    
                                # Get bounding box
                                x1, y1, x2, y2 = box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # Validate bounding box
                                if x1 >= x2 or y1 >= y2:
                                    continue
                                    
                                # Ensure box is within frame boundaries
                                x1 = max(0, min(x1, self.frame_width-1))
                                y1 = max(0, min(y1, self.frame_height-1))
                                x2 = max(0, min(x2, self.frame_width-1))
                                y2 = max(0, min(y2, self.frame_height-1))
                                
                                # Get confidence
                                conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                                
                                # Get face ID and person ID if previously associated
                                face_id = self.tracker_to_face_map.get(track_id, None)
                                person_id = self.tracker_to_person_map.get(track_id, None)
                                
                                # Apply Kalman filter to smooth bounding box if enabled
                                if self.use_kalman:
                                    try:
                                        bbox_array = np.array([x1, y1, x2, y2])
                                        # Update Kalman filter with current detection
                                        smoothed_bbox = self._update_kalman_filter(track_id, bbox_array)
                                        # Apply smoothed coordinates
                                        x1, y1, x2, y2 = [int(v) for v in smoothed_bbox]
                                    except Exception as e:
                                        print(f"Error applying Kalman filter: {e}")
                                
                                # Extract appearance features for this person (color histogram)
                                appearance = self._extract_appearance_features(frame, [x1, y1, x2, y2])
                                
                                # Calculate velocity if this is a previously tracked person
                                vx, vy = 0, 0
                                if track_id in self.tracked_persons and 'bbox' in self.tracked_persons[track_id]:
                                    prev_bbox = self.tracked_persons[track_id]['bbox']
                                    prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                                    prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                                    curr_cx = (x1 + x2) / 2
                                    curr_cy = (y1 + y2) / 2
                                    vx = curr_cx - prev_cx
                                    vy = curr_cy - prev_cy
                                    
                                    # Reject outliers for more robust velocity estimation
                                    # If velocity suddenly changes a lot, smooth it more aggressively
                                    if track_id in self.person_velocities:
                                        prev_vx, prev_vy = self.person_velocities[track_id]
                                        v_change = np.sqrt((vx - prev_vx)**2 + (vy - prev_vy)**2)
                                        
                                        # If velocity change is large, use more of the previous velocity
                                        if v_change > 10:
                                            # More aggressive smoothing (30% new, 70% old)
                                            vx = 0.3 * vx + 0.7 * prev_vx
                                            vy = 0.3 * vy + 0.7 * prev_vy
                                        else:
                                            # Normal smoothing (80% new, 20% old)
                                            vx = 0.8 * vx + 0.2 * prev_vx
                                            vy = 0.8 * vy + 0.2 * prev_vy
                                        
                                # Store updated velocity
                                self.person_velocities[track_id] = (vx, vy)
                                
                                # Update track history for this person
                                if track_id not in self.track_history:
                                    self.track_history[track_id] = []
                                
                                # Add current position to history
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                self.track_history[track_id].append((center_x, center_y, self.frame_count))
                                
                                # Keep only recent history
                                if len(self.track_history[track_id]) > self.max_track_history:
                                    self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
                                
                                # Determine occlusion status
                                occlusion_status = 'visible'
                                # Advanced occlusion detection will be handled in _handle_occlusions
                                
                                # Store the person
                                tracked_persons[track_id] = {
                                    'bbox': np.array([x1, y1, x2, y2]),
                                    'confidence': conf,
                                    'class_id': cls,
                                    'face_id': face_id,
                                    'person_id': person_id,
                                    'appearance': appearance,
                                    'velocity': (vx, vy),
                                    'last_seen': self.frame_count,
                                    'occlusion_status': occlusion_status
                                }
                            except Exception as e:
                                print(f"Error processing detection {i}: {e}")
                                continue
                except Exception as e:
                    print(f"Error processing YOLO results: {e}")
            
            # Check for occlusions and predict occluded person positions
            if self.actively_track_occlusions:
                try:
                    self._handle_occlusions(frame, tracked_persons, predicted_boxes)
                except Exception as e:
                    print(f"Error handling occlusions: {e}")
            
            # Update disappeared tracks
            try:
                self._update_disappeared_tracks(tracked_persons)
            except Exception as e:
                print(f"Error updating disappeared tracks: {e}")
            
            # Store current frame for next optical flow calculation
            self.last_frame = frame.copy()
            self.last_frame_gray = frame_gray
            
            # Try to reidentify new tracks with recently disappeared ones
            try:
                # Only process tracks that are new (not in previous frame)
                for track_id, person_data in tracked_persons.items():
                    if track_id not in self.tracked_persons and len(self.disappeared_tracks) > 0:
                        # Skip if already identified in this frame
                        if person_data.get('face_id') is None and person_data.get('person_id') is None:
                            self._try_reid_disappeared_track(track_id, person_data)
            except Exception as e:
                print(f"Error reidentifying new tracks: {e}")
            
            # Update stored tracked persons
            self.tracked_persons = tracked_persons
            
            return tracked_persons
        except Exception as e:
            print(f"Unexpected error in detect_and_track: {e}")
            return tracked_persons
    
    def draw_persons(self, frame, tracked_persons=None, show_person_id=True):
        """
        Draw bounding boxes and IDs on the detected persons.
        
        Args:
            frame (numpy.ndarray): Input image frame
            tracked_persons (dict, optional): Dictionary of tracked persons. If None, uses the current tracked persons.
            show_person_id (bool): Whether to show person IDs (from face recognition)
            
        Returns:
            numpy.ndarray: Frame with drawn person information
        """
        if tracked_persons is None:
            tracked_persons = self.tracked_persons
            
        if not tracked_persons:
            return frame
            
        result_frame = frame.copy()
        
        # Create a mapping of person IDs to their initial face IDs for consistent display
        person_to_initial_face = {}
        for track_id, person_data in tracked_persons.items():
            person_id = person_data.get('person_id', None)
            face_id = person_data.get('face_id', None)
            if person_id is not None and face_id is not None:
                if person_id in person_to_initial_face:
                    if face_id < person_to_initial_face[person_id]:
                        person_to_initial_face[person_id] = face_id
                else:
                    person_to_initial_face[person_id] = face_id
        
        try:
            # First draw trajectories for all tracked persons
            self._draw_trajectories(result_frame, tracked_persons)
        except Exception as e:
            print(f"Error drawing trajectories: {e}")
        
        # Then draw bounding boxes
        for track_id, person_data in tracked_persons.items():
            try:
                # Only draw if this person is currently tracked and has a valid bounding box
                if 'bbox' not in person_data:
                    continue
                    
                # Bounding box validation
                bbox = person_data['bbox']
                if bbox is None or len(bbox) != 4:
                    print(f"Invalid bbox for track_id {track_id}: {bbox}")
                    continue
                
                # Ensure the bounding box coordinates are valid integers and within frame boundaries
                h, w = frame.shape[:2]
                x1 = max(0, min(int(bbox[0]), w-1))
                y1 = max(0, min(int(bbox[1]), h-1))
                x2 = max(0, min(int(bbox[2]), w-1))
                y2 = max(0, min(int(bbox[3]), h-1))
                
                # Skip invalid bounding boxes
                if x1 >= x2 or y1 >= y2:
                    print(f"Invalid bbox dimensions for track_id {track_id}: [{x1},{y1},{x2},{y2}]")
                    continue
                
                # Get associated face ID and person ID if available
                face_id = person_data.get('face_id', None)
                person_id = person_data.get('person_id', None)
                
                # Use the initial face ID for this person for consistent display
                display_id = face_id
                if person_id is not None and person_id in person_to_initial_face:
                    display_id = person_to_initial_face[person_id]
                
                # Check if face is visible
                face_visible = True
                if 'face_data' in person_data and face_id is not None:
                    face_data = person_data.get('face_data', {})
                    face_visible = face_data.get('visible', True)
                    frames_since_seen = face_data.get('frames_since_seen', 0)
                    # Only consider the face to be visible if it's actually visible or very recently seen
                    face_visible = face_visible or frames_since_seen <= 3
                
                # Get occlusion status
                occlusion_status = person_data.get('occlusion_status', 'visible')
                
                # Adjust color based on identification and occlusion status
                if occlusion_status == 'fully_occluded':
                    # For occluded tracks, use dashed lines with identification color
                    if person_id is not None:
                        color = (0, 180, 0)  # Darker green for occluded identified person
                    elif face_id is not None:
                        color = (0, 180, 180)  # Darker yellow for occluded face detected
                    else:
                        color = (180, 0, 0)  # Darker blue for occluded unknown person
                    
                    # Draw dashed bounding box for occluded persons
                    self._draw_dashed_rectangle(result_frame, (x1, y1, x2, y2), color, thickness=2)
                else:
                    # For visible tracks, use solid lines with bright colors
                    if person_id is not None:
                        color = (0, 255, 0)  # Green - person identified
                    elif face_id is not None:
                        color = (0, 255, 255)  # Yellow - face detected but not identified
                    else:
                        color = (255, 0, 0)  # Blue - no face detected
                    
                    # Draw solid rectangle for visible persons
                    cv2.rectangle(
                        result_frame, 
                        (x1, y1), 
                        (x2, y2), 
                        color, 
                        2
                    )
                
                # Add confidence indicator (thickness of the box relates to confidence)
                confidence = person_data.get('confidence', 1.0)
                thickness = max(1, int(confidence * 3))
                
                # Draw tracking ID and face/person ID if available
                label_parts = []
                if show_person_id and person_id is not None:
                    if display_id is not None:
                        label_parts.append(f"Person: {person_id}")
                        # Only show face ID if different from person ID
                        if display_id != person_id:
                            label_parts.append(f"Face: {display_id}")
                    else:
                        label_parts.append(f"Person: {person_id}")
                elif face_id is not None:
                    label_parts.append(f"Face: {display_id}")
                    label_parts.append(f"Track: {track_id}")
                else:
                    label_parts.append(f"Track: {track_id}")
                
                # Add occlusion status for non-visible persons
                if occlusion_status != 'visible':
                    label_parts.append(f"({occlusion_status.replace('_', ' ')})")
                
                # Join all label parts
                label = " | ".join(label_parts)
                
                # Add background to text for better visibility
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)
                cv2.rectangle(
                    result_frame,
                    (x1, y1 - 20),
                    (x1 + text_size[0], y1),
                    (0, 0, 0),
                    -1
                )
                
                # Draw the label
                cv2.putText(
                    result_frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color, 
                    thickness
                )
                
                # Draw velocity vector if available
                if 'velocity' in person_data:
                    vx, vy = person_data['velocity']
                    # Only draw significant motion
                    if abs(vx) > 0.5 or abs(vy) > 0.5:
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        # Scale vector for visibility
                        end_x = int(center_x + vx * 3)
                        end_y = int(center_y + vy * 3)
                        
                        # Draw arrow showing velocity
                        cv2.arrowedLine(
                            result_frame,
                            (center_x, center_y),
                            (end_x, end_y),
                            (255, 0, 255),  # Magenta for velocity
                            thickness=2,
                            tipLength=0.3
                        )
            except Exception as e:
                print(f"Error drawing person {track_id}: {e}")
                continue
                    
        return result_frame
        
    def _draw_dashed_rectangle(self, image, bbox, color, thickness=1, dash_length=10):
        """
        Draw a dashed rectangle on an image.
        
        Args:
            image (numpy.ndarray): Image to draw on
            bbox (list or numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2]
            color (tuple): BGR color
            thickness (int): Line thickness
            dash_length (int): Length of each dash
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Draw top line
            for x in range(x1, x2, dash_length*2):
                x_end = min(x + dash_length, x2)
                cv2.line(image, (x, y1), (x_end, y1), color, thickness)
                
            # Draw bottom line
            for x in range(x1, x2, dash_length*2):
                x_end = min(x + dash_length, x2)
                cv2.line(image, (x, y2), (x_end, y2), color, thickness)
                
            # Draw left line
            for y in range(y1, y2, dash_length*2):
                y_end = min(y + dash_length, y2)
                cv2.line(image, (x1, y), (x1, y_end), color, thickness)
                
            # Draw right line
            for y in range(y1, y2, dash_length*2):
                y_end = min(y + dash_length, y2)
                cv2.line(image, (x2, y), (x2, y_end), color, thickness)
        except Exception as e:
            print(f"Error drawing dashed rectangle: {e}")
            
    def _draw_trajectories(self, image, tracked_persons):
        """
        Draw motion trajectories for tracked persons.
        
        Args:
            image (numpy.ndarray): Image to draw on
            tracked_persons (dict): Dictionary of tracked persons
        """
        for track_id, person_data in tracked_persons.items():
            try:
                # Skip if no trajectory data
                if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
                    continue
                    
                # Get trajectory points
                trajectory = self.track_history[track_id]
                
                # Get person color based on identification
                person_id = person_data.get('person_id', None)
                face_id = person_data.get('face_id', None)
                
                if person_id is not None:
                    color = (0, 255, 0)  # Green for identified person
                elif face_id is not None:
                    color = (0, 255, 255)  # Yellow for face detected
                else:
                    color = (255, 0, 0)  # Blue for unknown person
                    
                # Draw trajectory line
                for i in range(1, len(trajectory)):
                    try:
                        pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                        pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                        
                        # Calculate alpha (opacity) based on recency
                        current_frame = self.frame_count
                        alpha = min(1.0, 0.3 + 0.7 * (current_frame - trajectory[i-1][2]) / self.max_track_history)
                        
                        # Scale color by alpha for fading effect
                        scaled_color = tuple(int(c * alpha) for c in color)
                        
                        # Draw a line segment
                        cv2.line(image, pt1, pt2, scaled_color, thickness=2)
                    except Exception as e:
                        print(f"Error drawing trajectory segment: {e}")
                        continue
            except Exception as e:
                print(f"Error drawing trajectory for track {track_id}: {e}")
                continue
    
    def update_face_associations(self, tracked_persons, tracked_faces, face_module=None):
        """
        Associate tracked persons with detected faces based on IoU overlap.
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons with tracker IDs as keys
            tracked_faces (dict): Dictionary of tracked faces with face IDs as keys
            face_module (FaceModule, optional): Face module to get person IDs from face IDs
            
        Returns:
            dict: Updated tracked persons with face ID associations
        """
        if not tracked_persons or not tracked_faces:
            return tracked_persons
            
        # Create a mapping of person IDs to their initial face ID
        # This helps us maintain consistent face ID assignment
        person_to_initial_face = {}
        for face_id, face_data in tracked_faces.items():
            person_id = face_data.get('person_id', None)
            if person_id is not None:
                if person_id in person_to_initial_face:
                    if face_id < person_to_initial_face[person_id]:
                        person_to_initial_face[person_id] = face_id
                else:
                    person_to_initial_face[person_id] = face_id
            
        # For each person, find the best matching face based on IoU
        for track_id, person in tracked_persons.items():
            person_bbox = person['bbox']
            best_iou = 0
            best_face_id = None
            best_person_id = None
            best_face_data = None
            
            # If this tracker already has a person ID, try to use the initial face ID for that person
            current_person_id = person.get('person_id', None)
            if current_person_id is not None and current_person_id in person_to_initial_face:
                initial_face_id = person_to_initial_face[current_person_id]
                if initial_face_id in tracked_faces:
                    # Update the face ID to the initial one for consistency
                    best_face_id = initial_face_id
                    best_person_id = current_person_id
                    best_face_data = tracked_faces[initial_face_id]
            
            # If we don't have an initial face ID for this person yet, compute IoU with all faces
            if best_face_id is None:
                for face_id, face_data in tracked_faces.items():
                    face_bbox = face_data['bbox']
                    
                    # Calculate IoU between person and face bounding boxes
                    iou = self._calculate_iou(person_bbox, face_bbox)
                    
                    # Update best match if IoU is higher
                    if iou > best_iou and iou > 0.5:  # Threshold for considering a match
                        best_iou = iou
                        best_face_id = face_id
                        best_face_data = face_data
                        
                        # Get person ID from face if available
                        if 'person_id' in face_data:
                            best_person_id = face_data['person_id']
                        elif face_module is not None:
                            # Try to get person ID from face module
                            best_person_id = face_module.get_person_id_from_face_id(face_id)
            
            # Update person with face ID and person ID if found
            if best_face_id is not None:
                person['face_id'] = best_face_id
                self.tracker_to_face_map[track_id] = best_face_id
                
                # Store face visibility and other relevant data
                if best_face_data is not None:
                    person['face_data'] = {
                        'visible': best_face_data.get('visible', True),
                        'frames_since_seen': best_face_data.get('frames_since_seen', 0),
                        'first_seen': best_face_data.get('first_seen', 0),
                        'last_seen': best_face_data.get('last_seen', 0)
                    }
                
                if best_person_id is not None:
                    person['person_id'] = best_person_id
                    self.tracker_to_person_map[track_id] = best_person_id
                    
                    # Check if we found a face ID that doesn't match the initial face ID for this person
                    # If so, update our mapping to ensure consistency
                    if best_person_id in person_to_initial_face:
                        initial_face_id = person_to_initial_face[best_person_id]
                        if initial_face_id != best_face_id and initial_face_id < best_face_id:
                            # Use the smaller (earlier) face ID for this person
                            person['face_id'] = initial_face_id
                            self.tracker_to_face_map[track_id] = initial_face_id
                            # Update face data if available
                            if initial_face_id in tracked_faces:
                                face_data = tracked_faces[initial_face_id]
                                person['face_data'] = {
                                    'visible': face_data.get('visible', True),
                                    'frames_since_seen': face_data.get('frames_since_seen', 0),
                                    'first_seen': face_data.get('first_seen', 0),
                                    'last_seen': face_data.get('last_seen', 0)
                                }
                    else:
                        # This is a new person-face association, add to our mapping
                        person_to_initial_face[best_person_id] = best_face_id
        
        return tracked_persons
    
    def get_person_face_crops(self, frame, tracked_persons):
        """
        Get face region crops from tracked persons for face recognition.
        
        Args:
            frame (numpy.ndarray): Input image frame
            tracked_persons (dict): Dictionary of tracked persons
            
        Returns:
            dict: Dictionary mapping person track_id to face region crop
        """
        face_crops = {}
        
        # Validate input frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("Warning: Invalid frame passed to get_person_face_crops")
            return face_crops
        
        # Get frame dimensions for boundary checking
        frame_height, frame_width = frame.shape[:2]
        
        for track_id, person in tracked_persons.items():
            try:
                # Skip if this person already has a face ID
                if person.get('face_id') is not None:
                    continue
                
                # Validate person has a valid bbox
                if 'bbox' not in person or person['bbox'] is None:
                    continue
                
                bbox = person['bbox']
                
                # Ensure bbox is a valid array with 4 elements
                if not isinstance(bbox, (list, np.ndarray)) or len(bbox) != 4:
                    continue
                
                # Ensure all bbox values are numeric and not NaN
                try:
                    if any(not np.isfinite(coord) for coord in bbox):
                        continue
                except TypeError:
                    continue
                
                # Ensure bbox has valid dimensions (width and height > 0)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                
                # Extract the face region (upper portion of person bbox)
                # Assuming the face is in the top 1/3 of the person
                try:
                    face_height = max(1, int((bbox[3] - bbox[1]) // 3))
                    face_bbox = [
                        max(0, int(bbox[0])),
                        max(0, int(bbox[1])),
                        min(frame_width - 1, int(bbox[2])),
                        min(frame_height - 1, int(bbox[1] + face_height))
                    ]
                except (ValueError, TypeError) as e:
                    print(f"Error calculating face bbox for track {track_id}: {e}")
                    continue
                
                # Additional validation to ensure face_bbox has valid dimensions
                if face_bbox[0] >= face_bbox[2] or face_bbox[1] >= face_bbox[3]:
                    continue
                
                # Extract face region with robust error handling
                try:
                    face_img = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
                    
                    # Verify the crop has valid dimensions
                    if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                        continue
                    
                    # Store the crop with its bbox
                    face_crops[track_id] = {
                        'crop': face_img,
                        'bbox': np.array(face_bbox, dtype=np.int32)  # Ensure integer type
                    }
                except Exception as e:
                    print(f"Error extracting face crop for track {track_id}: {e}")
                    continue
                
            except Exception as e:
                print(f"Unexpected error processing track {track_id}: {e}")
                continue
            
        return face_crops 

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1 (numpy.ndarray): First bounding box [x1, y1, x2, y2]
            bbox2 (numpy.ndarray): Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate IoU
        union = bbox1_area + bbox2_area - intersection
        iou = intersection / union if union > 0 else 0.0
        
        return iou 

    def _extract_appearance_features(self, frame, bbox):
        """
        Extract appearance features (color histogram) for a person to aid in reidentification.
        Enhanced version with more robust color representation and spatial information.
        
        Args:
            frame (numpy.ndarray): Input image frame
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            dict: Appearance features
        """
        try:
            # Ensure bbox is within frame
            h, w = frame.shape[:2]
            x1 = max(0, min(int(bbox[0]), w-1))
            y1 = max(0, min(int(bbox[1]), h-1))
            x2 = max(0, min(int(bbox[2]), w-1))
            y2 = max(0, min(int(bbox[3]), h-1))
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            # Extract the person region
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None
                
            # Resize for consistency
            try:
                person_img = cv2.resize(person_img, (64, 128))
            except Exception as e:
                print(f"Error resizing person image: {e}")
                return None
            
            # Create a mask to focus on the person (simple approach)
            # This reduces background influence on the feature
            mask = np.ones_like(person_img[:,:,0], dtype=np.uint8) * 255
            
            # Divide person into regions for spatial information
            regions = []
            h_person, w_person = person_img.shape[:2]
            
            # Create 3 horizontal strips (head, torso, legs)
            head = person_img[0:h_person//3, :]
            torso = person_img[h_person//3:2*h_person//3, :]
            legs = person_img[2*h_person//3:, :]
            regions = [head, torso, legs]
            
            features = {}
            
            # Extract color histograms from different color spaces for better discrimination
            for i, region in enumerate(regions):
                region_name = ['head', 'torso', 'legs'][i]
                
                # Convert to different color spaces for richer representation
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                
                # HSV histograms (good for handling lighting changes)
                h_bins, s_bins, v_bins = 16, 16, 8
                h_ranges, s_ranges, v_ranges = [0, 180], [0, 256], [0, 256]
                
                # Calculate histograms for each channel
                h_hist = cv2.calcHist([hsv], [0], None, [h_bins], h_ranges)
                s_hist = cv2.calcHist([hsv], [1], None, [s_bins], s_ranges)
                v_hist = cv2.calcHist([hsv], [2], None, [v_bins], v_ranges)
                
                # Normalize histograms
                cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
                
                # LAB histograms (good for color perception)
                l_hist = cv2.calcHist([lab], [0], None, [16], [0, 256])
                a_hist = cv2.calcHist([lab], [1], None, [16], [0, 256])
                b_hist = cv2.calcHist([lab], [2], None, [16], [0, 256])
                
                cv2.normalize(l_hist, l_hist, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(a_hist, a_hist, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(b_hist, b_hist, 0, 1, cv2.NORM_MINMAX)
                
                # Store histograms
                features[f'{region_name}_h'] = h_hist.flatten()
                features[f'{region_name}_s'] = s_hist.flatten()
                features[f'{region_name}_v'] = v_hist.flatten()
                features[f'{region_name}_l'] = l_hist.flatten()
                features[f'{region_name}_a'] = a_hist.flatten()
                features[f'{region_name}_b'] = b_hist.flatten()
            
            # Calculate additional global features
            # Aspect ratio and size are important for person identification
            features['aspect_ratio'] = (x2 - x1) / (y2 - y1)
            features['height'] = y2 - y1
            features['area_ratio'] = (x2 - x1) * (y2 - y1) / (w * h)  # Proportion of frame occupied
            
            # Add dominant colors as a feature
            # Extract 3 dominant colors from the entire person
            Z = person_img.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 3  # Number of dominant colors
            try:
                _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                features['dominant_colors'] = centers.flatten()
            except Exception as e:
                print(f"Error extracting dominant colors: {e}")
            
            return features
        except Exception as e:
            print(f"Error extracting appearance features: {e}")
            return None
            
    def _compare_appearances(self, appearance1, appearance2):
        """
        Compare two appearance feature sets for similarity.
        Enhanced to handle the more detailed features from the improved extraction.
        
        Args:
            appearance1 (dict): First appearance features
            appearance2 (dict): Second appearance features
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if appearance1 is None or appearance2 is None:
            return 0.0
            
        try:
            # Get all common histogram feature keys
            histogram_keys = [k for k in appearance1.keys() if 
                             k in appearance2.keys() and 
                             k not in ['aspect_ratio', 'height', 'area_ratio', 'dominant_colors']]
            
            # Calculate histogram similarity
            hist_similarities = []
            for key in histogram_keys:
                if key in appearance1 and key in appearance2:
                    try:
                        # Use correlation method for comparing histograms
                        sim = cv2.compareHist(
                            np.array(appearance1[key], dtype=np.float32).reshape(-1, 1),
                            np.array(appearance2[key], dtype=np.float32).reshape(-1, 1),
                            cv2.HISTCMP_CORREL
                        )
                        hist_similarities.append(max(0, sim))  # Ensure non-negative
                    except Exception as e:
                        print(f"Error comparing histogram {key}: {e}")
            
            # Calculate histogram similarity as average of all histogram comparisons
            histogram_similarity = np.mean(hist_similarities) if hist_similarities else 0.0
            
            # Compare geometric features
            size_similarities = []
            
            # Aspect ratio similarity
            if 'aspect_ratio' in appearance1 and 'aspect_ratio' in appearance2:
                aspect1 = appearance1['aspect_ratio']
                aspect2 = appearance2['aspect_ratio']
                aspect_sim = min(aspect1, aspect2) / max(aspect1, aspect2)
                size_similarities.append(aspect_sim)
            
            # Height similarity
            if 'height' in appearance1 and 'height' in appearance2:
                height1 = appearance1['height']
                height2 = appearance2['height']
                height_sim = min(height1, height2) / max(height1, height2)
                size_similarities.append(height_sim)
            
            # Area ratio similarity
            if 'area_ratio' in appearance1 and 'area_ratio' in appearance2:
                area1 = appearance1['area_ratio']
                area2 = appearance2['area_ratio']
                area_sim = min(area1, area2) / max(area1, area2)
                size_similarities.append(area_sim)
            
            # Calculate size similarity as average of all size comparisons
            size_similarity = np.mean(size_similarities) if size_similarities else 0.0
            
            # Compare dominant colors if available
            color_similarity = 0.0
            if ('dominant_colors' in appearance1 and 'dominant_colors' in appearance2 and
                appearance1['dominant_colors'] is not None and appearance2['dominant_colors'] is not None):
                try:
                    colors1 = np.array(appearance1['dominant_colors']).reshape(-1, 3)
                    colors2 = np.array(appearance2['dominant_colors']).reshape(-1, 3)
                    
                    # Find minimum color distances between each pair
                    min_distances = []
                    for c1 in colors1:
                        dists = np.sqrt(np.sum((colors2 - c1)**2, axis=1))
                        min_distances.append(np.min(dists))
                    
                    # Calculate color similarity (invert and normalize distance)
                    avg_distance = np.mean(min_distances)
                    max_distance = 255 * np.sqrt(3)  # Maximum possible distance in RGB space
                    color_similarity = 1.0 - (avg_distance / max_distance)
                except Exception as e:
                    print(f"Error comparing dominant colors: {e}")
            
            # Calculate weighted similarity
            # Histograms are most important, then colors, then size
            weighted_similarity = (0.6 * histogram_similarity + 
                                  0.25 * color_similarity + 
                                  0.15 * size_similarity)
            
            return max(0.0, min(1.0, weighted_similarity))
        except Exception as e:
            print(f"Error comparing appearances: {e}")
            return 0.0
    
    def _update_disappeared_tracks(self, current_tracks):
        """
        Update the cache of disappeared tracks to help with reidentification.
        
        Args:
            current_tracks (dict): Currently active tracks
        """
        # Increment age for all disappeared tracks
        for track_id in list(self.disappeared_tracks.keys()):
            self.disappeared_tracks[track_id]['frames_since_seen'] += 1
            
            # Remove tracks that have been gone too long
            if self.disappeared_tracks[track_id]['frames_since_seen'] > self.max_disappearance_frames:
                del self.disappeared_tracks[track_id]
        
        # Add newly disappeared tracks to the cache
        for track_id in list(self.tracked_persons.keys()):
            if track_id not in current_tracks and track_id not in self.disappeared_tracks:
                if track_id in self.tracked_persons:
                    # This track has disappeared, add to cache
                    track_data = self.tracked_persons[track_id].copy()
                    track_data['frames_since_seen'] = 0
                    self.disappeared_tracks[track_id] = track_data
    
    def _handle_occlusions(self, frame, tracked_persons, predicted_boxes):
        """
        Handle occlusions by detecting overlaps and predicting occluded person positions.
        
        Args:
            frame (numpy.ndarray): Current frame
            tracked_persons (dict): Currently tracked persons
            predicted_boxes (dict): Predicted bounding boxes from motion model
        """
        try:
            # Create a map of all detected boxes for occlusion checking
            all_boxes = {track_id: person['bbox'] for track_id, person in tracked_persons.items() 
                        if 'bbox' in person and person['bbox'] is not None}
            
            # Get all person centers for distance-based occlusion detection
            person_centers = {}
            for track_id, person in tracked_persons.items():
                if 'bbox' in person and person['bbox'] is not None:
                    bbox = person['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    person_centers[track_id] = (center_x, center_y)
            
            # Check previously tracked persons that are missing in current frame
            for track_id, prev_person in self.tracked_persons.items():
                try:
                    # Skip if this person is already in current detections
                    if track_id in tracked_persons:
                        continue
                        
                    # Skip tracks without bbox or that disappeared too long ago
                    if 'bbox' not in prev_person or 'last_seen' not in prev_person:
                        continue
                        
                    frames_since_seen = self.frame_count - prev_person.get('last_seen', 0)
                    if frames_since_seen > 15:  # Only try to recover recent disappearances (increased from 10)
                        continue
                    
                    # Get predicted position from motion model or previous position
                    if track_id in predicted_boxes:
                        pred_bbox = predicted_boxes[track_id].copy()
                    else:
                        # Use Kalman filter prediction if available 
                        kalman_pred = None
                        if self.use_kalman:
                            kalman_pred = self._predict_kalman(track_id)
                            
                        if kalman_pred is not None:
                            pred_bbox = kalman_pred
                        else:
                            # Fall back to simple velocity prediction
                            # Use previous bbox and apply velocity if available
                            pred_bbox = prev_person['bbox'].copy()
                            if track_id in self.person_velocities:
                                vx, vy = self.person_velocities[track_id]
                                # Apply velocity with dampening based on frames since last seen
                                dampen_factor = max(0.1, 1.0 - 0.05 * frames_since_seen)
                                pred_bbox[0] += vx * dampen_factor
                                pred_bbox[1] += vy * dampen_factor
                                pred_bbox[2] += vx * dampen_factor
                                pred_bbox[3] += vy * dampen_factor
                    
                    # Calculate center of predicted bbox
                    pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
                    pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
                    
                    # Check if the predicted bbox is occluded by any current detection
                    is_occluded = False
                    occluding_tracks = []
                    
                    # First check using IoU for direct overlaps
                    for other_id, other_bbox in all_boxes.items():
                        iou = self._calculate_iou(pred_bbox, other_bbox)
                        if iou > self.occlusion_threshold:
                            is_occluded = True
                            occluding_tracks.append(other_id)
                    
                    # If not found by IoU, check using proximity between centers
                    if not is_occluded and person_centers:
                        for other_id, (cx, cy) in person_centers.items():
                            # Calculate distance between centers
                            distance = np.sqrt((pred_cx - cx)**2 + (pred_cy - cy)**2)
                            
                            # Check if distance is small enough to consider an occlusion
                            # Scale threshold based on person size
                            if track_id in self.tracked_persons and 'bbox' in self.tracked_persons[track_id]:
                                person_width = self.tracked_persons[track_id]['bbox'][2] - self.tracked_persons[track_id]['bbox'][0]
                                person_height = self.tracked_persons[track_id]['bbox'][3] - self.tracked_persons[track_id]['bbox'][1]
                                size_threshold = max(person_width, person_height) * 0.5
                                
                                if distance < size_threshold:
                                    is_occluded = True
                                    occluding_tracks.append(other_id)
                    
                    # Determine if this person is likely still in the frame
                    in_frame = self._is_within_bounds(pred_bbox, margin=0.05)
                    
                    # If this track is likely occluded or still in frame bounds, keep tracking it
                    if is_occluded or in_frame:
                        # Decay confidence based on how long since last seen
                        confidence_decay = max(0.3, 1.0 - 0.05 * frames_since_seen)
                        base_confidence = prev_person.get('confidence', 0.5)
                        
                        # Carry forward the track with predicted position
                        appearance = prev_person.get('appearance', None)
                        face_id = prev_person.get('face_id', None)
                        person_id = prev_person.get('person_id', None)
                        
                        # Get velocity with decay
                        vx, vy = self.person_velocities.get(track_id, (0, 0))
                        
                        # Decay velocity for occluded tracks
                        vx *= 0.95  # Slower decay for more persistent prediction
                        vy *= 0.95
                        self.person_velocities[track_id] = (vx, vy)
                        
                        # Determine occlusion status
                        if is_occluded:
                            occlusion_status = 'fully_occluded'
                        else:
                            occlusion_status = 'out_of_frame'
                        
                        # Add to tracked_persons with occluded status
                        tracked_persons[track_id] = {
                            'bbox': np.array(pred_bbox),
                            'confidence': base_confidence * confidence_decay,
                            'class_id': prev_person.get('class_id', self.person_class_id),
                            'face_id': face_id,
                            'person_id': person_id,
                            'appearance': appearance,
                            'velocity': (vx, vy),
                            'last_seen': prev_person.get('last_seen', self.frame_count - frames_since_seen),
                            'frames_since_seen': frames_since_seen,
                            'occlusion_status': occlusion_status,
                            'occluded_by': occluding_tracks if is_occluded else []
                        }
                        
                        # Update track history even for occluded tracks
                        if track_id in self.track_history:
                            self.track_history[track_id].append((pred_cx, pred_cy, self.frame_count))
                            if len(self.track_history[track_id]) > self.max_track_history:
                                self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
                except Exception as e:
                    print(f"Error handling occlusion for track {track_id}: {e}")
        except Exception as e:
            print(f"Error in _handle_occlusions: {e}")

    def _try_reid_disappeared_track(self, track_id, track_data):
        """
        Try to re-identify a new track with a recently disappeared track.
        This enhanced version uses multiple features for more robust matching.
        
        Args:
            track_id (int): ID of the new track
            track_data (dict): Data for the new track
        """
        try:
            # Skip if this track already has face/person identification
            if track_data.get('face_id') is not None or track_data.get('person_id') is not None:
                return
            
            best_similarity = 0.45  # Slightly increased threshold for more reliable matching
            best_match_id = None
            best_score = 0.0
            
            # Get current track position and dimensions
            if 'bbox' not in track_data or track_data['bbox'] is None:
                return
            
            curr_bbox = track_data['bbox']
            curr_x = (curr_bbox[0] + curr_bbox[2]) / 2
            curr_y = (curr_bbox[1] + curr_bbox[3]) / 2
            curr_width = curr_bbox[2] - curr_bbox[0]
            curr_height = curr_bbox[3] - curr_bbox[1]
            
            # Compare this track with all disappeared tracks
            for old_id, old_data in self.disappeared_tracks.items():
                try:
                    total_score = 0.0
                    feature_weights = {
                        'appearance': 0.5,   # 50% weight to appearance
                        'position': 0.25,    # 25% weight to position
                        'size': 0.15,        # 15% weight to size
                        'velocity': 0.1      # 10% weight to velocity
                    }
                    used_features = {}
                    
                    # Skip if disappeared track has been gone too long
                    frames_gone = old_data.get('frames_since_seen', 0)
                    if frames_gone > min(30, self.max_disappearance_frames / 3):  # More strict for reidentification
                        continue
                    
                    # 1. Compare appearances (if available)
                    if ('appearance' in old_data and old_data['appearance'] is not None and
                        'appearance' in track_data and track_data['appearance'] is not None):
                        try:
                            appearance_sim = self._compare_appearances(track_data['appearance'], old_data['appearance'])
                            used_features['appearance'] = appearance_sim
                        except Exception as e:
                            print(f"Error comparing appearances: {e}")
                    
                    # 2. Compare position and motion
                    if 'bbox' in old_data:
                        try:
                            old_bbox = old_data['bbox']
                            old_x = (old_bbox[0] + old_bbox[2]) / 2
                            old_y = (old_bbox[1] + old_bbox[3]) / 2
                            
                            # Account for motion
                            if 'velocity' in old_data:
                                vx, vy = old_data['velocity']
                                # Predict where the disappeared track should be now
                                old_x += vx * frames_gone
                                old_y += vy * frames_gone
                            
                            # Calculate distance (scaled by frame size for normalization)
                            distance = np.sqrt((curr_x - old_x)**2 + (curr_y - old_y)**2)
                            max_distance = np.sqrt(self.frame_width**2 + self.frame_height**2) / 4
                            position_sim = 1.0 - min(1.0, distance / max_distance)
                            
                            used_features['position'] = position_sim
                        except Exception as e:
                            print(f"Error comparing position: {e}")
                    
                    # 3. Compare size and aspect ratio
                    if 'bbox' in old_data:
                        try:
                            old_width = old_bbox[2] - old_bbox[0]
                            old_height = old_bbox[3] - old_bbox[1]
                            
                            # Size similarity
                            width_ratio = min(old_width, curr_width) / max(old_width, curr_width)
                            height_ratio = min(old_height, curr_height) / max(old_height, curr_height)
                            
                            # Aspect ratio similarity
                            old_aspect = old_width / old_height if old_height > 0 else 1.0
                            curr_aspect = curr_width / curr_height if curr_height > 0 else 1.0
                            aspect_ratio = min(old_aspect, curr_aspect) / max(old_aspect, curr_aspect)
                            
                            # Combined size similarity
                            size_sim = (width_ratio + height_ratio + aspect_ratio) / 3
                            used_features['size'] = size_sim
                        except Exception as e:
                            print(f"Error comparing size: {e}")
                    
                    # 4. Compare velocity directions (if available)
                    if 'velocity' in old_data and 'velocity' in track_data:
                        try:
                            old_vx, old_vy = old_data['velocity']
                            curr_vx, curr_vy = track_data['velocity']
                            
                            # Only compare if both have significant velocity
                            old_v_mag = np.sqrt(old_vx**2 + old_vy**2)
                            curr_v_mag = np.sqrt(curr_vx**2 + curr_vy**2)
                            
                            if old_v_mag > 1.0 and curr_v_mag > 1.0:
                                # Calculate cosine similarity between velocity vectors
                                dot_product = old_vx * curr_vx + old_vy * curr_vy
                                velocity_sim = max(0, (dot_product / (old_v_mag * curr_v_mag) + 1) / 2)
                                used_features['velocity'] = velocity_sim
                        except Exception as e:
                            print(f"Error comparing velocity: {e}")
                    
                    # Calculate weighted score
                    if used_features:
                        for feature, value in used_features.items():
                            total_score += value * feature_weights[feature]
                        
                        # Normalize by weights of used features
                        used_weights_sum = sum(feature_weights[f] for f in used_features.keys())
                        final_score = total_score / used_weights_sum if used_weights_sum > 0 else 0
                        
                        # Reduce score based on time disappeared (fresher matches are better)
                        time_discount = max(0.7, 1.0 - frames_gone * 0.01)
                        final_score *= time_discount
                        
                        # If this is the best match so far, remember it
                        if final_score > best_score:
                            best_score = final_score
                            best_match_id = old_id
                except Exception as e:
                    print(f"Error evaluating match for track {old_id}: {e}")
                    continue
            
            # If we found a good match, transfer identity information
            if best_match_id is not None and best_score > best_similarity:
                try:
                    print(f"Reidentified track {track_id} as previous track {best_match_id} with score {best_score:.2f}")
                    
                    # Transfer face_id and person_id if available
                    if 'face_id' in self.disappeared_tracks[best_match_id] and self.disappeared_tracks[best_match_id]['face_id'] is not None:
                        track_data['face_id'] = self.disappeared_tracks[best_match_id]['face_id']
                        self.tracker_to_face_map[track_id] = track_data['face_id']
                        
                    if 'person_id' in self.disappeared_tracks[best_match_id] and self.disappeared_tracks[best_match_id]['person_id'] is not None:
                        track_data['person_id'] = self.disappeared_tracks[best_match_id]['person_id']
                        self.tracker_to_person_map[track_id] = track_data['person_id']
                        
                    # Transfer face_data if available
                    if 'face_data' in self.disappeared_tracks[best_match_id]:
                        track_data['face_data'] = self.disappeared_tracks[best_match_id]['face_data'].copy()
                        
                    # Copy tracking history
                    if best_match_id in self.track_history:
                        self.track_history[track_id] = self.track_history[best_match_id].copy()
                        # Add current position
                        bbox = track_data['bbox']
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        self.track_history[track_id].append((center_x, center_y, self.frame_count))
                        # Keep only recent history
                        if len(self.track_history[track_id]) > self.max_track_history:
                            self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
                        
                    # Delete the matched disappeared track
                    del self.disappeared_tracks[best_match_id]
                except Exception as e:
                    print(f"Error transferring identity information: {e}")
        except Exception as e:
            print(f"Error in _try_reid_disappeared_track: {e}")
    
    def _is_within_bounds(self, bbox, margin=0.1):
        """
        Check if a bounding box is within the frame bounds with a margin.
        
        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            margin (float): Margin factor for frame (0.1 = 10% of frame width/height)
            
        Returns:
            bool: True if within extended bounds
        """
        # Define extended boundaries with margin
        left_bound = -self.frame_width * margin
        top_bound = -self.frame_height * margin
        right_bound = self.frame_width * (1 + margin)
        bottom_bound = self.frame_height * (1 + margin)
        
        # Check if bbox center is within bounds
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        return (left_bound < center_x < right_bound and 
                top_bound < center_y < bottom_bound) 

    def _init_kalman_filter(self, track_id, bbox):
        """
        Initialize a Kalman filter for a new track.
        
        Args:
            track_id (int): Track ID
            bbox (numpy.ndarray): Initial bounding box [x1, y1, x2, y2]
        """
        try:
            # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
            # 8 state variables, 4 for bbox coordinates and 4 for velocities
            kalman = cv2.KalmanFilter(8, 4)
            
            # Measurement matrix (maps state to measurement)
            kalman.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]
            ], np.float32)
            
            # Transition matrix (how state evolves)
            # x1' = x1 + vx1, y1' = y1 + vy1, etc.
            kalman.transitionMatrix = np.array([
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ], np.float32)
            
            # Process noise covariance matrix (Q)
            # Represents noise in the process
            process_noise_scale = 0.03
            kalman.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise_scale
            
            # Measurement noise covariance matrix (R)
            # Represents noise in the measurement
            measurement_noise_scale = 0.1
            kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise_scale
            
            # Error covariance matrix (P)
            kalman.errorCovPost = np.eye(8, dtype=np.float32)
            
            # Initial state
            kalman.statePost = np.array([
                [bbox[0]],  # x1
                [bbox[1]],  # y1
                [bbox[2]],  # x2
                [bbox[3]],  # y2
                [0],        # vx1
                [0],        # vy1
                [0],        # vx2
                [0]         # vy2
            ], np.float32)
            
            self.kalman_filters[track_id] = kalman
            self.kalman_states[track_id] = kalman.statePost.copy()
        except Exception as e:
            print(f"Error initializing Kalman filter for track {track_id}: {e}")

    def _update_kalman_filter(self, track_id, bbox):
        """
        Update the Kalman filter for a track with a new measurement.
        
        Args:
            track_id (int): Track ID
            bbox (numpy.ndarray): New bounding box measurement [x1, y1, x2, y2]
            
        Returns:
            numpy.ndarray: Corrected bounding box
        """
        try:
            if track_id not in self.kalman_filters:
                self._init_kalman_filter(track_id, bbox)
                return bbox
            
            kalman = self.kalman_filters[track_id]
            
            # Predict new state
            predicted = kalman.predict()
            
            # Ensure predicted bbox is valid
            if predicted[0] > predicted[2]:
                predicted[2] = predicted[0] + 1
            if predicted[1] > predicted[3]:
                predicted[3] = predicted[1] + 1
            
            # Update with measurement
            measurement = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], np.float32)
            corrected = kalman.correct(measurement)
            
            # Store updated state
            self.kalman_states[track_id] = corrected.copy()
            
            # Return corrected bbox
            return np.array([
                corrected[0][0],
                corrected[1][0],
                corrected[2][0],
                corrected[3][0]
            ])
        except Exception as e:
            print(f"Error updating Kalman filter for track {track_id}: {e}")
            return bbox

    def _predict_kalman(self, track_id):
        """
        Use Kalman filter to predict the next position of a track.
        
        Args:
            track_id (int): Track ID
            
        Returns:
            numpy.ndarray: Predicted bounding box or None if filter not found
        """
        try:
            if track_id not in self.kalman_filters:
                return None
            
            kalman = self.kalman_filters[track_id]
            
            # Predict without updating
            predicted = kalman.predict().copy()
            
            # Revert to previous state (since we're just predicting, not updating)
            kalman.statePost = self.kalman_states[track_id].copy()
            
            # Ensure predicted bbox is valid
            if predicted[0] > predicted[2]:
                predicted[2] = predicted[0] + 1
            if predicted[1] > predicted[3]:
                predicted[3] = predicted[1] + 1
            
            # Return predicted bbox
            return np.array([
                predicted[0][0],
                predicted[1][0],
                predicted[2][0],
                predicted[3][0]
            ])
        except Exception as e:
            print(f"Error predicting with Kalman filter for track {track_id}: {e}")
            return None 