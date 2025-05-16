import numpy as np
import cv2

class OcclusionHandler:
    """
    Utility class for detecting and handling occlusions in person tracking
    """
    def __init__(self, config=None):
        """
        Initialize the occlusion handler
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        
        # Overlap threshold to detect occlusion
        self.overlap_threshold = self.config.get('overlap_threshold', 0.5)
        
        # Minimum area ratio for occlusion detection
        self.min_area_ratio = self.config.get('min_area_ratio', 0.3)
        
        # History of track positions for occlusion reasoning
        self.track_history = {}  # track_id -> list of positions
        
        # Maximum history length
        self.max_history = self.config.get('max_history', 20)
        
        # Occlusion status for each track
        self.occlusion_status = {}  # track_id -> status
        
    def detect_occlusions(self, tracked_persons):
        """
        Detect occlusions between tracked persons
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons
            
        Returns:
            dict: Updated tracked_persons with occlusion information
        """
        # Update position history for each track
        self._update_position_history(tracked_persons)
        
        # Get all bounding boxes
        bboxes = []
        track_ids = []
        
        for track_id, person in tracked_persons.items():
            if 'bbox' in person:
                bboxes.append(person['bbox'])
                track_ids.append(track_id)
        
        # Check for overlaps between all pairs
        num_tracks = len(track_ids)
        
        # Clear previous statuses
        for track_id in tracked_persons:
            if track_id not in self.occlusion_status:
                self.occlusion_status[track_id] = 'visible'
        
        # Check for occlusions
        for i in range(num_tracks):
            track_id_i = track_ids[i]
            bbox_i = bboxes[i]
            
            # Default status is visible
            occlusion_status = 'visible'
            occluded_by = []
            
            for j in range(num_tracks):
                if i == j:
                    continue
                    
                track_id_j = track_ids[j]
                bbox_j = bboxes[j]
                
                # Calculate overlap
                iou = self._calculate_iou(bbox_i, bbox_j)
                
                # Calculate area ratio (for detecting smaller/larger objects)
                area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
                area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                
                # Check if i is occluded by j
                if iou > self.overlap_threshold and area_i < area_j:
                    # Calculate overlap percentage
                    overlap_area = self._calculate_overlap_area(bbox_i, bbox_j)
                    overlap_percentage = overlap_area / area_i
                    
                    # If significant overlap, consider it occluded
                    if overlap_percentage > self.min_area_ratio:
                        occlusion_status = 'partially_occluded'
                        occluded_by.append(track_id_j)
                        
                        # If very high overlap, consider it fully occluded
                        if overlap_percentage > 0.7:
                            occlusion_status = 'fully_occluded'
            
            # Update occlusion status
            self.occlusion_status[track_id_i] = occlusion_status
            
            # Update person data
            tracked_persons[track_id_i]['occlusion_status'] = occlusion_status
            if occluded_by:
                tracked_persons[track_id_i]['occluded_by'] = occluded_by
        
        return tracked_persons
    
    def _update_position_history(self, tracked_persons):
        """
        Update position history for each track
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons
        """
        for track_id, person in tracked_persons.items():
            if 'bbox' not in person:
                continue
                
            bbox = person['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                
            self.track_history[track_id].append((center_x, center_y))
            
            # Limit history length
            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id] = self.track_history[track_id][-self.max_history:]
    
    def predict_occluded_positions(self, tracked_persons, frame_shape):
        """
        Predict positions for occluded tracks
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons
            frame_shape (tuple): Frame dimensions (height, width)
            
        Returns:
            dict: Updated tracked_persons with predicted positions for occluded tracks
        """
        frame_height, frame_width = frame_shape[:2]
        
        # Process each track
        for track_id, person in tracked_persons.items():
            # Skip if not occluded or no history
            if (track_id not in self.occlusion_status or
                self.occlusion_status[track_id] == 'visible' or
                track_id not in self.track_history or
                len(self.track_history[track_id]) < 3):
                continue
            
            # Get position history
            history = self.track_history[track_id]
            
            # Calculate velocity from recent positions
            recent_positions = history[-3:]
            
            # Calculate average velocity
            velocities = []
            for i in range(1, len(recent_positions)):
                vx = recent_positions[i][0] - recent_positions[i-1][0]
                vy = recent_positions[i][1] - recent_positions[i-1][1]
                velocities.append((vx, vy))
            
            avg_vx = sum(v[0] for v in velocities) / len(velocities)
            avg_vy = sum(v[1] for v in velocities) / len(velocities)
            
            # Predict new position
            last_pos = recent_positions[-1]
            predicted_x = last_pos[0] + avg_vx
            predicted_y = last_pos[1] + avg_vy
            
            # Constrain to frame boundaries
            predicted_x = max(0, min(frame_width, predicted_x))
            predicted_y = max(0, min(frame_height, predicted_y))
            
            # Store predicted position
            person['predicted_position'] = (predicted_x, predicted_y)
            
            # If we have bbox, create predicted bbox with same dimensions
            if 'bbox' in person:
                bbox = person['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                predicted_bbox = [
                    predicted_x - width/2,
                    predicted_y - height/2,
                    predicted_x + width/2,
                    predicted_y + height/2
                ]
                
                # Constrain to frame boundaries
                predicted_bbox[0] = max(0, predicted_bbox[0])
                predicted_bbox[1] = max(0, predicted_bbox[1])
                predicted_bbox[2] = min(frame_width, predicted_bbox[2])
                predicted_bbox[3] = min(frame_height, predicted_bbox[3])
                
                person['predicted_bbox'] = predicted_bbox
        
        return tracked_persons
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1, bbox2 (list): Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # If the boxes don't intersect, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate each box area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate union area
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area
        
        return iou
    
    def _calculate_overlap_area(self, bbox1, bbox2):
        """
        Calculate overlap area between two bounding boxes
        
        Args:
            bbox1, bbox2 (list): Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            float: Overlap area
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # If the boxes don't intersect, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        return intersection_area
    
    def draw_occlusion_visualization(self, frame, tracked_persons):
        """
        Draw occlusion visualization on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            tracked_persons (dict): Dictionary of tracked persons
            
        Returns:
            numpy.ndarray: Frame with occlusion visualization
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
            
        result_frame = frame.copy()
        
        for track_id, person in tracked_persons.items():
            try:
                if 'occlusion_status' not in person or 'bbox' not in person:
                    continue
                    
                if person.get('bbox') is None:
                    continue
                
                status = person['occlusion_status']
                bbox = person['bbox']
                
                # Ensure bbox is valid
                try:
                    if isinstance(bbox, np.ndarray):
                        bbox_array = bbox
                    else:
                        bbox_array = np.array(bbox)
                        
                    if len(bbox_array) != 4:
                        continue
                        
                    x1, y1, x2, y2 = [int(c) for c in bbox_array]
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Invalid bbox format for track {track_id}: {e}")
                    continue
                
                # Validate coordinates are within frame bounds
                h, w = result_frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Draw bounding box with color based on occlusion status
                if status == 'visible':
                    color = (0, 255, 0)  # Green
                elif status == 'partially_occluded':
                    color = (0, 165, 255)  # Orange
                else:  # fully_occluded
                    color = (0, 0, 255)  # Red
                
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw status text
                cv2.putText(
                    result_frame,
                    status,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                
                # Draw predicted position if occluded
                if status != 'visible' and 'predicted_bbox' in person and person['predicted_bbox'] is not None:
                    try:
                        pred_bbox = person['predicted_bbox']
                        
                        # Validate predicted bbox
                        if len(pred_bbox) != 4:
                            continue
                            
                        px1, py1, px2, py2 = [int(c) for c in pred_bbox]
                        
                        # Validate coordinates are within frame bounds
                        px1 = max(0, min(px1, w-1))
                        py1 = max(0, min(py1, h-1))
                        px2 = max(0, min(px2, w-1))
                        py2 = max(0, min(py2, h-1))
                        
                        if px1 >= px2 or py1 >= py2:
                            continue
                        
                        # Draw dashed box for prediction
                        self._draw_dashed_rectangle(
                            result_frame,
                            (px1, py1),
                            (px2, py2),
                            color
                        )
                        
                        # Draw line from current to predicted
                        center_current = ((x1 + x2) // 2, (y1 + y2) // 2)
                        center_predicted = ((px1 + px2) // 2, (py1 + py2) // 2)
                        
                        cv2.line(
                            result_frame,
                            center_current,
                            center_predicted,
                            color,
                            1,
                            cv2.LINE_AA
                        )
                    except (ValueError, TypeError, IndexError) as e:
                        print(f"Error drawing predicted bbox for track {track_id}: {e}")
                        continue
            except Exception as e:
                print(f"Error drawing occlusion visualization for track {track_id}: {e}")
        
        return result_frame
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1, dash_length=8):
        """
        Draw a dashed rectangle on an image
        
        Args:
            img (numpy.ndarray): Image to draw on
            pt1 (tuple): Top-left corner of rectangle
            pt2 (tuple): Bottom-right corner of rectangle
            color (tuple): BGR color
            thickness (int): Line thickness
            dash_length (int): Length of dashes
        """
        # Draw horizontal dashed lines
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top line
        for x in range(x1, x2, dash_length * 2):
            x_end = min(x + dash_length, x2)
            cv2.line(img, (x, y1), (x_end, y1), color, thickness)
            
        # Bottom line
        for x in range(x1, x2, dash_length * 2):
            x_end = min(x + dash_length, x2)
            cv2.line(img, (x, y2), (x_end, y2), color, thickness)
            
        # Left line
        for y in range(y1, y2, dash_length * 2):
            y_end = min(y + dash_length, y2)
            cv2.line(img, (x1, y), (x1, y_end), color, thickness)
            
        # Right line
        for y in range(y1, y2, dash_length * 2):
            y_end = min(y + dash_length, y2)
            cv2.line(img, (x2, y), (x2, y_end), color, thickness) 