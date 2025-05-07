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
        self.device = 'cuda:0'  # Force CUDA device

        # Check if GPU is available
        try:
            import torch
            if torch.cuda.is_available():
                print("Using GPU acceleration for person tracking")
            else:
                raise RuntimeError("GPU acceleration required but CUDA is not available")
        except ImportError:
            raise ImportError("PyTorch not available. Please install PyTorch with CUDA support")

        # Load YOLO model
        model_path = config.get('model_path', 'yolov8n.pt')
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Person class ID
        self.person_class_id = 0

        # Tracking storage
        self.tracked_persons = {}
        self.tracker_to_face_map = {}
        self.tracker_to_person_map = {}

        # Appearance & reID
        self.person_appearance_features = {}
        self.disappeared_tracks = {}
        self.max_disappearance_frames = 120

        # Velocity & Kalman
        self.person_velocities = {}
        self.use_kalman = config.get('use_kalman', True)
        self.kalman_filters = {}
        self.kalman_states = {}

        # Deep features
        self.use_deep_features = False
        try:
            cv2.dnn.readNet
            self.use_deep_features = config.get('use_deep_features', True)
            if self.use_deep_features:
                print("Using deep appearance features for better reidentification")
        except (AttributeError, ImportError):
            print("OpenCV DNN module not available, using basic appearance features")

        # Tracking metrics & history
        self.track_history = {}
        self.max_track_history = 30
        self.track_qualities = {}
        self.track_lifetimes = {}
        self.min_quality_threshold = config.get('min_quality_threshold', 0.3)

        # Occlusion handling
        self.occlusion_threshold = config.get('occlusion_threshold', 0.4)
        self.actively_track_occlusions = config.get('track_occlusions', True)
        self.occlusion_graph = {}
        self.occlusion_state_history = {}
        self.occlusion_recovery_buffer = 30

        # Spatial reasoning
        self.use_spatial_reasoning = config.get('use_spatial_reasoning', True)
        self.spatial_relationships = {}
        self.depth_ordering = []
        self.min_overlap_iou = config.get('min_overlap_iou', 0.1)

        # Trajectory prediction
        self.use_advanced_trajectory = config.get('use_advanced_trajectory', True)
        self.trajectory_models = {}
        self.trajectory_history_length = config.get('trajectory_history_length', 50)

        # Identity management
        self.person_identity_features = {}
        self.identity_confusion_matrix = {}
        self.last_person_positions = {}
        self.identity_consistency_threshold = config.get('identity_consistency_threshold', 0.6)
        self.enable_strict_identity_checking = config.get('enable_strict_identity_checking', True)
        self.identity_feature_history = {}
        self.max_feature_history = 10
        self.next_generated_id = 1000
        self.used_person_ids = set()

        # Frame storage
        self.frame_width = 1920
        self.frame_height = 1080
        self.frame_count = 0
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
        tracked_persons = {}
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("Warning: Invalid frame passed to detect_and_track")
            return tracked_persons

        # Update frame dimensions & count
        h, w = frame.shape[:2]
        self.frame_height, self.frame_width = h, w
        self.frame_count += 1

        # Grayscale for optical flow
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error converting frame to grayscale: {e}")
            frame_gray = None

        # Optical flow vectors
        flow_vectors = {}
        if self.last_frame_gray is not None and frame_gray is not None and self.actively_track_occlusions:
            try:
                for tid, person in self.tracked_persons.items():
                    bbox = person.get('bbox')
                    if bbox is None or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    points = []
                    step = max(5, min((x2-x1)//4, (y2-y1)//4))
                    for x in range(x1, x2, step):
                        for y in range(y1, y2, step):
                            if 0 <= x < w and 0 <= y < h:
                                points.append([float(x), float(y)])
                    points.append([float(cx), float(cy)])
                    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.last_frame_gray, frame_gray, pts, None,
                        winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    good_old = pts[status.flatten()==1].reshape(-1, 2)
                    good_new = new_pts[status.flatten()==1].reshape(-1, 2)
                    if len(good_old) > 0:
                        dx, dy = np.mean(good_new - good_old, axis=0)
                        flow_vectors[tid] = (dx, dy)
            except Exception as e:
                print(f"Error calculating optical flow: {e}")

        # Run YOLO tracking
        try:
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                persist=True,
                tracker="botsort.yaml",
                device=self.device
            )
        except Exception as e:
            print(f"Error running YOLO model: {e}")
            results = None

        # Predict boxes using flow + velocity
        predicted_boxes = {}
        if self.use_advanced_trajectory:
            for tid, person in self.tracked_persons.items():
                bbox = person.get('bbox')
                if bbox is None:
                    continue
                dx, dy = flow_vectors.get(tid, (0, 0))
                vx, vy = self.person_velocities.get(tid, (0, 0))
                mag_f = np.hypot(dx, dy)
                mag_v = np.hypot(vx, vy)
                if mag_f < 30 and abs(mag_f - mag_v) < 20:
                    pdx = 0.7 * dx + 0.3 * vx
                    pdy = 0.7 * dy + 0.3 * vy
                else:
                    pdx, pdy = vx, vy
                x1, y1, x2, y2 = bbox
                pb = [
                    max(0, min(x1 + pdx, w-1)),
                    max(0, min(y1 + pdy, h-1)),
                    max(0, min(x2 + pdx, w-1)),
                    max(0, min(y2 + pdy, h-1))
                ]
                predicted_boxes[tid] = pb

        # Process detections
        if results and len(results) > 0:
            det = results[0]
            boxes = getattr(det, 'boxes', [])
            if boxes is not None:
                self._update_track_qualities()
                for box in boxes:
                    cls = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                    if cls != self.person_class_id:
                        continue
                    tid = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if x1 >= x2 or y1 >= y2:
                        continue
                    x1, x2 = max(0, x1), min(w-1, x2)
                    y1, y2 = max(0, y1), min(h-1, y2)
                    conf = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                    face_id = self.tracker_to_face_map.get(tid)
                    person_id = self.tracker_to_person_map.get(tid)
                    if self.use_kalman:
                        try:
                            sm = self._update_kalman_filter(tid, np.array([x1, y1, x2, y2]))
                            x1, y1, x2, y2 = map(int, sm)
                        except Exception as e:
                            print(f"Error applying Kalman filter: {e}")
                    appearance = self._extract_appearance_features(frame, [x1, y1, x2, y2])
                    vx, vy = 0, 0
                    if tid in self.tracked_persons:
                        prev_bbox = self.tracked_persons[tid]['bbox']
                        prev_cx, prev_cy = (prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2
                        curr_cx, curr_cy = (x1 + x2) / 2, (y1 + y2) / 2
                        vx, vy = curr_cx - prev_cx, curr_cy - prev_cy
                        if tid in self.person_velocities:
                            pvx, pvy = self.person_velocities[tid]
                            vchg = np.hypot(vx - pvx, vy - pvy)
                            if vchg > 10:
                                vx = 0.3 * vx + 0.7 * pvx
                                vy = 0.3 * vy + 0.7 * pvy
                            else:
                                vx = 0.8 * vx + 0.2 * pvx
                                vy = 0.8 * vy + 0.2 * pvy
                    self.person_velocities[tid] = (vx, vy)
                    hist = self.track_history.setdefault(tid, [])
                    hist.append(((x1 + x2) / 2, (y1 + y2) / 2, self.frame_count))
                    if len(hist) > self.max_track_history:
                        hist[:] = hist[-self.max_track_history:]
                    tracked_persons[tid] = {
                        'bbox': np.array([x1, y1, x2, y2]),
                        'confidence': conf,
                        'class_id': cls,
                        'face_id': face_id,
                        'person_id': person_id,
                        'appearance': appearance,
                        'velocity': (vx, vy),
                        'last_seen': self.frame_count,
                        'occlusion_status': 'visible'
                    }
                    self._update_track_quality(tid, tracked_persons[tid])

        # Trajectory & occlusion post-processing
        if self.use_advanced_trajectory:
            try:
                self._predict_trajectories(tracked_persons)
            except Exception:
                pass
        try:
            self._analyze_occlusions(tracked_persons)
        except Exception:
            pass
        if self.actively_track_occlusions:
            try:
                self._handle_occlusions(frame, tracked_persons, predicted_boxes)
            except Exception:
                pass
        self._update_disappeared_tracks(tracked_persons)
        self.last_frame, self.last_frame_gray = frame.copy(), frame_gray

        # Re-identify disappeared tracks
        for tid, data in tracked_persons.items():
            if tid not in self.tracked_persons:
                try:
                    self._try_reid_disappeared_track(tid, data)
                except Exception:
                    pass

        self.tracked_persons = tracked_persons
        return tracked_persons

    def draw_persons(self, frame, tracked_persons=None, show_person_id=True):
        """
        Draw bounding boxes, trajectories, and IDs on the frame.
        """
        if tracked_persons is None:
            tracked_persons = self.tracked_persons
        if not tracked_persons:
            return frame
        out = frame.copy()

        # Initial face map for label consistency
        person_to_initial_face = {}
        for tid, pd in tracked_persons.items():
            pid, fid = pd.get('person_id'), pd.get('face_id')
            if pid is not None and fid is not None:
                person_to_initial_face[pid] = min(fid, person_to_initial_face.get(pid, fid))

        try:
            self._draw_trajectories(out, tracked_persons)
            if self.use_advanced_trajectory:
                self._draw_predicted_trajectories(out, tracked_persons)
        except Exception as e:
            print(f"Error drawing trajectories: {e}")

        for tid, pd in tracked_persons.items():
            try:
                bb = pd.get('bbox')
                if bb is None or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bb)
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w-1, x2)
                y1, y2 = max(0, y1), min(h-1, y2)
                if x1 >= x2 or y1 >= y2:
                    continue

                pid, fid = pd.get('person_id'), pd.get('face_id')
                if pid is not None:
                    color = (0, 255, 0)
                elif fid is not None:
                    color = (0, 255, 255)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                thickness = max(1, int(pd.get('confidence', 1.0) * 3))

                # Build label
                display_id = fid if pid is None else person_to_initial_face.get(pid, fid)
                parts = []
                if show_person_id and pid is not None:
                    parts.append(f"Person: {pid}")
                    if display_id is not None and display_id != pid:
                        parts.append(f"Face: {display_id}")
                elif fid is not None:
                    parts.append(f"Face: {display_id}")
                    parts.append(f"Track: {tid}")
                else:
                    parts.append(f"Track: {tid}")

                oc = pd.get('occlusion_status', 'visible')
                if oc != 'visible':
                    parts.append(f"({oc.replace('_', ' ')})")

                label = " | ".join(parts)
                ts, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)
                cv2.rectangle(out, (x1, y1-20), (x1+ts[0], y1), (0, 0, 0), -1)
                cv2.putText(out, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

                # Velocity arrow
                vx, vy = pd.get('velocity', (0, 0))
                if abs(vx) > 0.5 or abs(vy) > 0.5:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    ex, ey = int(cx + vx*3), int(cy + vy*3)
                    cv2.arrowedLine(out, (cx, cy), (ex, ey), (255, 0, 255), 2, tipLength=0.3)
            except Exception as e:
                print(f"Error drawing person {tid}: {e}")
        return out

    def update_face_associations(self, tracked_persons, tracked_faces, face_module=None):
        """
        Associate tracked persons with tracked faces based on IoU and consistency checks.
        """
        if not tracked_persons or not tracked_faces:
            return tracked_persons
        try:
            # Initial face per person
            person_to_initial_face = {}
            for fid, fd in tracked_faces.items():
                pid = fd.get('person_id')
                if pid is not None:
                    person_to_initial_face[pid] = min(fid, person_to_initial_face.get(pid, fid))
                    self.used_person_ids.add(pid)

            # Compute IoU overlap scores
            overlap_scores = []
            for tid, pd in tracked_persons.items():
                pb = pd['bbox']
                for fid, fd in tracked_faces.items():
                    fb = fd['bbox']
                    iou = self._calculate_iou(pb, fb)
                    if iou > 0.3:
                        overlap_scores.append((tid, fid, iou))
            overlap_scores.sort(key=lambda x: x[2], reverse=True)

            assigned, rev = {}, {}
            for tid, fid, iou in overlap_scores:
                if tid not in assigned and fid not in rev:
                    assigned[tid] = fid
                    rev[fid] = tid

            # Assign and consistency checks
            for tid, pd in tracked_persons.items():
                prev_fid = pd.get('face_id')
                prev_pid = pd.get('person_id')
                if tid in assigned:
                    fid = assigned[tid]
                    new_pid = tracked_faces[fid].get('person_id')
                    if prev_pid is not None and new_pid is not None and prev_pid != new_pid and self.enable_strict_identity_checking:
                        old_feats = self.person_identity_features.get(tid)
                        new_feats = self.person_identity_features.get(new_pid)
                        sim = self._compare_identity_features(old_feats, new_feats)
                        if sim < self.identity_consistency_threshold:
                            # Reject
                            conflict = (prev_pid, new_pid)
                            self.identity_confusion_matrix[conflict] = self.identity_confusion_matrix.get(conflict, 0) + 1
                            pd['face_id'] = prev_fid
                            pd['person_id'] = prev_pid
                            continue
                    pd['face_id'] = fid
                    self.tracker_to_face_map[tid] = fid
                    if new_pid is not None:
                        pd['person_id'] = new_pid
                        self.tracker_to_person_map[tid] = new_pid
                        self._update_identity_features(tid, pd, new_pid)
                        self.used_person_ids.add(new_pid)
                elif prev_fid is not None and prev_fid in tracked_faces and prev_fid not in rev:
                    fb = tracked_faces[prev_fid]['bbox']
                    pb = pd['bbox']
                    fc = ((fb[0]+fb[2])/2, (fb[1]+fb[3])/2)
                    pc = ((pb[0]+pb[2])/2, (pb[1]+pb[3])/2)
                    dist = np.hypot(fc[0]-pc[0], fc[1]-pc[1])
                    height = pb[3] - pb[1]
                    if dist < height * 0.5:
                        pd['face_id'] = prev_fid
                        rev[prev_fid] = tid

            if self.enable_strict_identity_checking:
                self._resolve_identity_conflicts(tracked_persons)
            return tracked_persons
        except Exception as e:
            print(f"Error in update_face_associations: {e}")
            return tracked_persons
    
    def _compare_identity_features(self, features1, features2):
        """
        Compare two sets of identity features for similarity.
        
        Args:
            features1 (dict): First feature set
            features2 (dict): Second feature set
            
        Returns:
            float: Similarity score (0-1)
        """
        try:
            if features1 is None or features2 is None:
                return 0.0
                
            scores = []
            
            # Compare appearances using our existing method
            if ('appearance' in features1 and 'appearance' in features2 and
                features1['appearance'] is not None and features2['appearance'] is not None):
                app_score = self._compare_appearances(features1['appearance'], features2['appearance'])
                scores.append(app_score * 0.6)  # Appearance is most important
            
            # Compare height
            if ('height' in features1 and 'height' in features2 and
                features1['height'] is not None and features2['height'] is not None):
                height1 = features1['height']
                height2 = features2['height']
                height_ratio = min(height1, height2) / max(height1, height2)
                scores.append(height_ratio * 0.2)
            
            # Compare aspect ratio
            if ('aspect_ratio' in features1 and 'aspect_ratio' in features2 and
                features1['aspect_ratio'] is not None and features2['aspect_ratio'] is not None):
                ar1 = features1['aspect_ratio']
                ar2 = features2['aspect_ratio']
                ar_ratio = min(ar1, ar2) / max(ar1, ar2)
                scores.append(ar_ratio * 0.2)
            
            # Calculate overall similarity
            if not scores:
                return 0.0
                
            return sum(scores) / sum(s > 0 for s in scores)
        except Exception as e:
            print(f"Error comparing identity features: {e}")
            return 0.0
    
    def _resolve_identity_conflicts(self, tracked_persons):
        """
        Resolve any remaining identity conflicts by ensuring unique person IDs.
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons
        """
        try:
            # First, identify all currently active person IDs
            active_person_ids = {}
            for track_id, person in tracked_persons.items():
                person_id = person.get('person_id')
                if person_id is not None:
                    if person_id not in active_person_ids:
                        active_person_ids[person_id] = [track_id]
                    else:
                        active_person_ids[person_id].append(track_id)
            
            # For each person ID that appears multiple times, resolve conflicts
            for person_id, track_ids in active_person_ids.items():
                if len(track_ids) > 1:
                    print(f"Detected ID conflict: Person ID {person_id} assigned to {len(track_ids)} tracks")
                    
                    # Find the track with the highest quality/confidence
                    best_track_id = None
                    best_quality = -1
                    
                    for track_id in track_ids:
                        person = tracked_persons[track_id]
                        # Use track quality if available, otherwise use detection confidence
                        quality = self.track_qualities.get(track_id, {}).get('overall', 0)
                        confidence = person.get('confidence', 0)
                        combined_score = quality * 0.7 + confidence * 0.3
                        
                        if combined_score > best_quality:
                            best_quality = combined_score
                            best_track_id = track_id
                    
                    # The best track keeps the original ID, others get new IDs
                    for track_id in track_ids:
                        if track_id != best_track_id:
                            person = tracked_persons[track_id]
                            # Generate a new unique ID
                            new_id = self._generate_unique_person_id()
                            
                            print(f"Reassigning track {track_id} from person ID {person_id} to {new_id}")
                            
                            # Update with new ID
                            person['person_id'] = new_id
                            self.tracker_to_person_map[track_id] = new_id
                            
                            # Clone identity features if they exist
                            if person_id in self.person_identity_features:
                                self.person_identity_features[new_id] = self.person_identity_features[person_id].copy()
        except Exception as e:
            print(f"Error resolving identity conflicts: {e}")
    
    def _generate_unique_person_id(self):
        """
        Generate a unique person ID that hasn't been used before.
        
        Returns:
            int: New unique person ID
        """
        # Start from our next available ID
        new_id = self.next_generated_id
        
        # Find an ID that's not in use
        while new_id in self.used_person_ids:
            new_id += 1
        
        # Update next ID and mark as used
        self.next_generated_id = new_id + 1
        self.used_person_ids.add(new_id)
        
        return new_id
    
                    
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

                # Validate bbox
                if 'bbox' not in person or person['bbox'] is None:
                    continue
                bbox = person['bbox']

                # Check bbox format
                if not isinstance(bbox, (list, np.ndarray)) or len(bbox) != 4:
                    continue

                # Check numeric
                try:
                    if any(not np.isfinite(coord) for coord in bbox):
                        continue
                except TypeError:
                    continue

                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue

                # Define face region (top 1/3)
                face_height = max(1, int((y2 - y1) // 3))
                fx1 = max(0, int(x1))
                fy1 = max(0, int(y1))
                fx2 = min(frame_width - 1, int(x2))
                fy2 = min(frame_height - 1, int(y1 + face_height))

                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                face_img = frame[fy1:fy2, fx1:fx2]
                if face_img.size == 0:
                    continue

                face_crops[track_id] = {
                    'crop': face_img,
                    'bbox': np.array([fx1, fy1, fx2, fy2], dtype=np.int32)
                }

            except Exception as e:
                print(f"Error extracting face crop for track {track_id}: {e}")
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
        Enhanced to use the occlusion graph for more accurate prediction of occluded persons.
        
        Args:
            frame (numpy.ndarray): Current frame
            tracked_persons (dict): Currently tracked persons
            predicted_boxes (dict): Predicted bounding boxes from motion model
        """
        try:
            # Extract occluded persons from the occlusion graph
            occluded_persons = set()
            for occluder_id, occluded_ids in self.occlusion_graph.items():
                for occluded_id in occluded_ids:
                    occluded_persons.add(occluded_id)
            
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
                    
                    # Use occlusion graph to determine if this person is likely occluded
                    is_known_occluded = track_id in occluded_persons
                    
                    # For known occluded persons, we keep them longer
                    max_frames_to_keep = self.occlusion_recovery_buffer if is_known_occluded else 15
                    
                    if frames_since_seen > max_frames_to_keep:
                        continue
                    
                    # Check track quality to decide whether to keep predicting
                    track_quality = self.track_qualities.get(track_id, {}).get('overall', 0.5)
                    
                    # We're more lenient with high-quality tracks and known occluded persons
                    quality_threshold = self.min_quality_threshold * 0.8 if is_known_occluded else self.min_quality_threshold
                    if track_quality < quality_threshold and frames_since_seen > 5 and not is_known_occluded:
                        # Low quality tracks are dropped more quickly
                        continue
                    
                    # Get best predicted position using our advanced trajectory models
                    pred_bbox = None
                    
                    # Try advanced trajectory prediction first (most accurate)
                    if self.use_advanced_trajectory and track_id in self.trajectory_models:
                        model = self.trajectory_models[track_id]
                        
                        # Select the best prediction model based on track quality and trajectory characteristics
                        predictions = model.get('predictions', {})
                        
                        if predictions:
                            # For known occluded persons, prefer motion-aware predictions
                            if is_known_occluded and track_id in self.spatial_relationships:
                                # Get IDs of people occluding this person
                                occluders = self.spatial_relationships[track_id].get('behind', [])
                                
                                # If we have occluders with velocity, adjust prediction based on their motion
                                if occluders and any(occ_id in self.person_velocities for occ_id in occluders):
                                    # Use occluder's velocity to help predict the occluded person's movement
                                    for occ_id in occluders:
                                        if occ_id in tracked_persons and occ_id in self.person_velocities:
                                            # Get occluder's velocity
                                            occ_vx, occ_vy = self.person_velocities[occ_id]
                                            
                                            # If the occluder is moving significantly, adjust prediction
                                            if abs(occ_vx) > 2 or abs(occ_vy) > 2:
                                                # Get last known position
                                                bbox = prev_person['bbox']
                                                
                                                # Apply occluder's velocity with dampening
                                                dampening = 0.7  # Assume occluded person moves somewhat with occluder
                                                pred_bbox = [
                                                    bbox[0] + occ_vx * dampening,
                                                    bbox[1] + occ_vy * dampening,
                                                    bbox[2] + occ_vx * dampening,
                                                    bbox[3] + occ_vy * dampening
                                                ]
                                                break  # Use first significant occluder
                        
                            # If no special occlusion handling applied, use standard prediction methods
                            if pred_bbox is None:
                                # Determine which prediction method to use based on track quality and trajectory complexity
                                prediction_method = 'linear'  # Default to linear
                                
                                # Use curvature to determine path complexity
                                curvatures = model.get('curvatures', [])
                                avg_curvature = np.mean(curvatures) if curvatures else 0
                                
                                # If high quality track with significant curvature, use curve prediction
                                if track_quality > 0.7 and avg_curvature > 0.2 and 'curve' in predictions:
                                    prediction_method = 'curve'
                                # If moderate quality with some acceleration, use quadratic
                                elif track_quality > 0.5 and 'quadratic' in predictions:
                                    prediction_method = 'quadratic'
                                # Otherwise use linear for simplicity and reliability
                                
                                # Get prediction for current frame
                                frame_offset = min(frames_since_seen, len(predictions.get(prediction_method, [])) - 1)
                                if frame_offset >= 0 and prediction_method in predictions and len(predictions[prediction_method]) > frame_offset:
                                    pred_x, pred_y, pred_w, pred_h = predictions[prediction_method][frame_offset]
                                    
                                    # Convert center and dimensions to bbox format
                                    pred_bbox = [
                                        pred_x - pred_w/2,  # x1
                                        pred_y - pred_h/2,  # y1
                                        pred_x + pred_w/2,  # x2
                                        pred_y + pred_h/2   # y2
                                    ]
                
                    # If no advanced prediction, try Kalman filter
                    if pred_bbox is None and self.use_kalman:
                        kalman_pred = self._predict_kalman(track_id)
                        if kalman_pred is not None:
                            pred_bbox = kalman_pred
                    
                    # If still no prediction, fall back to simpler velocity-based prediction
                    if pred_bbox is None:
                        if track_id in predicted_boxes:
                            pred_bbox = predicted_boxes[track_id].copy()
                        else:
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
                    occlusion_confidence = 0.0
                    
                    # For known occluded persons from the occlusion graph, force occlusion status
                    if is_known_occluded:
                        is_occluded = True
                        occlusion_confidence = 0.8
                        
                        # Find occluders from the graph
                        for occluder_id, occluded_ids in self.occlusion_graph.items():
                            if track_id in occluded_ids and occluder_id in tracked_persons:
                                occluding_tracks.append(occluder_id)
                    else:
                        # Standard occlusion detection for non-graph-tracked occlusions
                        # First check using IoU for direct overlaps
                        for other_id, other_bbox in all_boxes.items():
                            iou = self._calculate_iou(pred_bbox, other_bbox)
                            if iou > self.occlusion_threshold:
                                is_occluded = True
                                occluding_tracks.append(other_id)
                                occlusion_confidence = max(occlusion_confidence, iou)
                        
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
                                        occlusion_confidence = max(occlusion_confidence, 1.0 - (distance / size_threshold))
                    
                    # Determine if this person is likely still in the frame
                    in_frame = self._is_within_bounds(pred_bbox, margin=0.05)
                    
                    # If this track is likely occluded or still in frame bounds, keep tracking it
                    if is_occluded or in_frame:
                        # Use track quality to determine base confidence
                        base_confidence = max(0.3, track_quality) if track_id in self.track_qualities else 0.5
                        
                        # Decay confidence based on how long since last seen
                        confidence_decay = max(0.3, 1.0 - 0.05 * frames_since_seen)
                        
                        # Boost confidence for known occluded persons
                        if is_known_occluded:
                            confidence_decay *= 1.3  # 30% boost for known occlusions
                        
                        # Boost confidence if we're using advanced trajectory prediction
                        if self.use_advanced_trajectory and track_id in self.trajectory_models:
                            model = self.trajectory_models[track_id]
                            if model.get('predictions'):
                                confidence_decay *= 1.2  # 20% boost for advanced prediction
                        
                        # Final confidence is product of base and decay
                        final_confidence = base_confidence * confidence_decay
                        
                        # Carry forward the track with predicted position
                        appearance = prev_person.get('appearance', None)
                        face_id = prev_person.get('face_id', None)
                        person_id = prev_person.get('person_id', None)
                        
                        # Get velocity with decay
                        vx, vy = self.person_velocities.get(track_id, (0, 0))
                        
                        # Decay velocity - slower decay for known occlusions
                        vx_decay = 0.98 if is_known_occluded else 0.95 
                        vy_decay = 0.98 if is_known_occluded else 0.95
                        vx *= vx_decay
                        vy *= vy_decay
                        self.person_velocities[track_id] = (vx, vy)
                        
                        # Determine occlusion status
                        if is_occluded:
                            occlusion_status = 'fully_occluded' if occlusion_confidence > 0.7 else 'partially_occluded'
                        else:
                            occlusion_status = 'out_of_frame'
                        
                        # Set confidence based on occlusion relationship consistency
                        if is_known_occluded:
                            for occluder_id in occluding_tracks:
                                occlusion_key = (occluder_id, track_id)
                                if occlusion_key in self.occlusion_state_history:
                                    history = self.occlusion_state_history[occlusion_key]
                                    # Longer occlusion relationships get more confidence
                                    if history['duration'] > 5:
                                        final_confidence *= 1.1  # Another 10% boost for consistent occlusions
                        
                        # Add to tracked_persons with occluded status
                        tracked_persons[track_id] = {
                            'bbox': np.array(pred_bbox),
                            'confidence': final_confidence,
                            'class_id': prev_person.get('class_id', self.person_class_id),
                            'face_id': face_id,
                            'person_id': person_id,
                            'appearance': appearance,
                            'velocity': (vx, vy),
                            'last_seen': prev_person.get('last_seen', self.frame_count - frames_since_seen),
                            'frames_since_seen': frames_since_seen,
                            'occlusion_status': occlusion_status,
                            'occluded_by': occluding_tracks if is_occluded else [],
                            'track_quality': track_quality,
                            'known_occluded': is_known_occluded  # Flag for special handling in visualization
                        }
                        
                        # Update track history even for occluded tracks
                        if track_id in self.track_history:
                            self.track_history[track_id].append((pred_cx, pred_cy, self.frame_count))
                            if len(self.track_history[track_id]) > self.max_track_history:
                                self.track_history[track_id] = self.track_history[track_id][-self.max_track_history:]
                except Exception as e:
                    print(f"Error handling occlusion for track {track_id}: {e}")
                    continue  # Skip this track and continue with others
        except Exception as e:
            print(f"Error in _handle_occlusions for track {track_id}: {e}")
            # Skip this track and continue with others
            tracked_persons[track_id] = prev_person
        except Exception as e:
            print(f"Error in _handle_occlusions: {e}")
            return tracked_persons  # Return what we have so far

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

    def _predict_trajectories(self, tracked_persons):
        """
        Predict future trajectories for all tracked persons using multiple models.
        This improves prediction accuracy by combining different prediction methods.
        
        Args:
            tracked_persons (dict): Currently tracked persons
        """
        # Skip if no tracks to process
        if not tracked_persons:
            return
        
        # Update trajectory models for each person
        for track_id, person in tracked_persons.items():
            try:
                if 'bbox' not in person or person['bbox'] is None:
                    continue
                    
                bbox = person['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                # Initialize trajectory model if this is a new track
                if track_id not in self.trajectory_models:
                    self.trajectory_models[track_id] = {
                        'positions': [],
                        'timestamps': [],
                        'velocities': [],
                        'accelerations': [],
                        'curvatures': [],
                        'predictions': {},
                        'last_updated': self.frame_count
                    }
                
                model = self.trajectory_models[track_id]
                
                # Add current position to trajectory history
                model['positions'].append((cx, cy, w, h))
                model['timestamps'].append(self.frame_count)
                
                # Keep only recent history
                if len(model['positions']) > self.trajectory_history_length:
                    model['positions'] = model['positions'][-self.trajectory_history_length:]
                    model['timestamps'] = model['timestamps'][-self.trajectory_history_length:]
                
                # Need at least 3 points for velocity and acceleration calculation
                if len(model['positions']) >= 3:
                    # Calculate velocities (first derivative of position)
                    positions = np.array(model['positions'])
                    times = np.array(model['timestamps'])
                    
                    # Velocities: change in position over time
                    # Using finite difference for approximation
                    velocities = []
                    for i in range(1, len(positions)):
                        dt = max(1, times[i] - times[i-1])  # Time difference (at least 1 frame)
                        dx = positions[i][0] - positions[i-1][0]  # Change in x
                        dy = positions[i][1] - positions[i-1][1]  # Change in y
                        velocities.append((dx/dt, dy/dt))
                    
                    model['velocities'] = velocities
                    
                    # Accelerations: change in velocity over time (second derivative)
                    # Need at least 2 velocity measurements
                    if len(velocities) >= 2:
                        accelerations = []
                        for i in range(1, len(velocities)):
                            dt = max(1, times[i] - times[i-1])
                            dvx = velocities[i][0] - velocities[i-1][0]
                            dvy = velocities[i][1] - velocities[i-1][1]
                            accelerations.append((dvx/dt, dvy/dt))
                        
                        model['accelerations'] = accelerations
                    
                    # Calculate curvature (how much the path is bending)
                    if len(positions) >= 3:
                        curvatures = []
                        for i in range(1, len(positions)-1):
                            # Get three consecutive points
                            p1 = positions[i-1][:2]  # Only x,y
                            p2 = positions[i][:2]
                            p3 = positions[i+1][:2]
                            
                            # Calculate vectors
                            v1 = np.array(p2) - np.array(p1)
                            v2 = np.array(p3) - np.array(p2)
                            
                            # Calculate angle between vectors (in radians)
                            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                                dot_product = np.dot(v1, v2)
                                magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
                                angle = np.arccos(np.clip(dot_product / magnitudes, -1.0, 1.0))
                                curvatures.append(angle)
                            else:
                                curvatures.append(0)
                        
                        model['curvatures'] = curvatures
                    
                    # Make predictions using multiple methods
                    predictions = {}
                    
                    # 1. Linear prediction (extrapolation of current velocity)
                    if len(velocities) > 0:
                        last_vx, last_vy = velocities[-1]
                        last_x, last_y, last_w, last_h = positions[-1]
                        
                        # Predict next 15 positions (0.5 second at 30 FPS)
                        linear_predictions = []
                        for i in range(1, 16):
                            pred_x = last_x + last_vx * i
                            pred_y = last_y + last_vy * i
                            linear_predictions.append((pred_x, pred_y, last_w, last_h))
                        
                        predictions['linear'] = linear_predictions
                    
                    # 2. Quadratic prediction (accounts for acceleration)
                    if len(model['accelerations']) > 0:
                        last_ax, last_ay = model['accelerations'][-1]
                        last_vx, last_vy = velocities[-1]
                        last_x, last_y, last_w, last_h = positions[-1]
                        
                        quadratic_predictions = []
                        for i in range(1, 16):
                            # s = s0 + v0*t + 0.5*a*t^2
                            pred_x = last_x + last_vx * i + 0.5 * last_ax * i * i
                            pred_y = last_y + last_vy * i + 0.5 * last_ay * i * i
                            quadratic_predictions.append((pred_x, pred_y, last_w, last_h))
                        
                        predictions['quadratic'] = quadratic_predictions
                    
                    # 3. Curve fitting prediction (for complex trajectories)
                    if len(positions) >= 5:  # Need enough points for curve fitting
                        try:
                            # Extract x and y coordinates
                            xs = [p[0] for p in positions[-10:]]  # Last 10 positions
                            ys = [p[1] for p in positions[-10:]]
                            ts = [t - times[-10] for t in times[-10:]]  # Relative times
                            
                            # Fit polynomial for x and y separately
                            # Higher degree for more complex paths, lower for simpler ones
                            # Use track quality to determine polynomial degree
                            quality = self.track_qualities.get(track_id, {}).get('overall', 0.5)
                            degree = 2 if quality > 0.7 else 1  # More complex model for high-quality tracks
                            
                            if len(xs) > degree:  # Need more points than polynomial degree
                                x_poly = np.polyfit(ts, xs, degree)
                                y_poly = np.polyfit(ts, ys, degree)
                                
                                # Generate predictions
                                curve_predictions = []
                                last_w, last_h = positions[-1][2], positions[-1][3]
                                
                                for i in range(1, 16):
                                    t = ts[-1] + i  # Future time
                                    pred_x = np.polyval(x_poly, t)
                                    pred_y = np.polyval(y_poly, t)
                                    curve_predictions.append((pred_x, pred_y, last_w, last_h))
                                
                                predictions['curve'] = curve_predictions
                        except Exception as e:
                            print(f"Error in curve fitting prediction: {e}")
                    
                    # Store all predictions
                    model['predictions'] = predictions
                    model['last_updated'] = self.frame_count
            except Exception as e:
                print(f"Error updating trajectory model for track {track_id}: {e}")
                continue
            
    def _update_track_qualities(self):
        """
        Update quality metrics for all tracks to assess tracking reliability.
        This helps identify and handle unreliable tracks.
        """
        # Skip if no tracks exist
        if not self.tracked_persons:
            return
            
        # Go through all existing tracks
        for track_id, person in self.tracked_persons.items():
            try:
                # Initialize quality metrics if this is a new track
                if track_id not in self.track_qualities:
                    self.track_qualities[track_id] = {
                        'detection_confidence': 0.0,
                        'temporal_consistency': 0.0,
                        'velocity_stability': 0.0,
                        'appearance_consistency': 0.0,
                        'overall': 0.0,
                        'history': []
                    }
            except Exception as e:
                print(f"Error initializing track quality for track {track_id}: {e}")
                
    def _update_track_quality(self, track_id, person_data):
        """
        Update quality metrics for a specific track.
        
        Args:
            track_id (int): Track ID
            person_data (dict): Current track data
        """
        try:
            # Initialize if not already present
            if track_id not in self.track_qualities:
                self.track_qualities[track_id] = {
                    'detection_confidence': 0.0,
                    'temporal_consistency': 0.0,
                    'velocity_stability': 0.0,
                    'appearance_consistency': 0.0,
                    'overall': 0.0,
                    'history': []
                }
                
            quality = self.track_qualities[track_id]
            
            # 1. Detection confidence (from detector)
            current_conf = person_data.get('confidence', 0.5)
            quality['detection_confidence'] = current_conf
            
            # 2. Temporal consistency (regular detections over time)
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                # Calculate time gaps between detections
                times = [pos[2] for pos in self.track_history[track_id]]
                gaps = np.diff(times)
                avg_gap = np.mean(gaps)
                consistency = max(0, min(1, 1.0 - (avg_gap - 1) / 10))  # Normalize: 1 is perfect, >10 is poor
                quality['temporal_consistency'] = consistency
            else:
                quality['temporal_consistency'] = 0.5  # Default for new tracks
            
            # 3. Velocity stability (consistent motion)
            if track_id in self.person_velocities:
                if 'history' in quality and len(quality['history']) > 0:
                    # Compare current velocity with historical average
                    vx, vy = self.person_velocities[track_id]
                    v_magnitude = np.sqrt(vx*vx + vy*vy)
                    
                    # Get velocity history
                    v_history = [entry.get('velocity_magnitude', 0) for entry in quality['history'][-5:]]
                    if v_history:
                        # Calculate stability as inverse of velocity variance
                        avg_v = np.mean(v_history)
                        if avg_v > 0:
                            variance = np.mean([(v - avg_v)**2 for v in v_history + [v_magnitude]])
                            stability = max(0, min(1, 1.0 - variance / (avg_v + 1e-5)))
                            quality['velocity_stability'] = stability
            
            # 4. Appearance consistency
            if 'appearance' in person_data and person_data['appearance'] is not None:
                if 'history' in quality and len(quality['history']) > 0:
                    # Get most recent appearance
                    recent_appearances = [
                        entry.get('appearance') for entry in quality['history'][-5:]
                        if 'appearance' in entry and entry['appearance'] is not None
                    ]
                    
                    if recent_appearances:
                        # Calculate average appearance similarity
                        similarities = []
                        for prev_appearance in recent_appearances:
                            try:
                                sim = self._compare_appearances(person_data['appearance'], prev_appearance)
                                similarities.append(sim)
                            except Exception:
                                pass
                        
                        if similarities:
                            avg_similarity = np.mean(similarities)
                            quality['appearance_consistency'] = avg_similarity
            
            # Update history with current information
            quality['history'].append({
                'frame': self.frame_count,
                'confidence': current_conf,
                'velocity_magnitude': np.sqrt(person_data['velocity'][0]**2 + person_data['velocity'][1]**2)
                    if 'velocity' in person_data else 0,
                'appearance': person_data.get('appearance')
            })
            
            # Keep history manageable
            if len(quality['history']) > 20:
                quality['history'] = quality['history'][-20:]
            
            # Calculate overall quality
            weights = {
                'detection_confidence': 0.3,
                'temporal_consistency': 0.2,
                'velocity_stability': 0.2,
                'appearance_consistency': 0.3
            }
            
            overall = sum(
                quality[metric] * weight
                for metric, weight in weights.items()
                if quality[metric] is not None
            )
            
            # Store overall quality
            quality['overall'] = overall
        except Exception as e:
            print(f"Error updating track quality for track {track_id}: {e}")
    
    def _draw_predicted_trajectories(self, image, tracked_persons):
        """
        Draw predicted future trajectories for occluded or partially visible persons.
        
        Args:
            image (numpy.ndarray): Image to draw on
            tracked_persons (dict): Dictionary of tracked persons
        """
        try:
            for track_id, person_data in tracked_persons.items():
                # Only draw predictions for occluded tracks or those with uncertain detection
                occlusion_status = person_data.get('occlusion_status', 'visible')
                confidence = person_data.get('confidence', 1.0)
                
                if (occlusion_status != 'visible' or confidence < 0.7) and track_id in self.trajectory_models:
                    model = self.trajectory_models[track_id]
                    predictions = model.get('predictions', {})
                    
                    if not predictions:
                        continue
                    
                    # Determine best prediction method
                    # For visualization, use curve if available, then quadratic, then linear
                    method = None
                    for preferred in ['curve', 'quadratic', 'linear']:
                        if preferred in predictions:
                            method = preferred
                            break
                        
                    if method is None:
                        continue
                    
                    # Get prediction
                    future_points = predictions[method]
                    
                    if not future_points:
                        continue
                    
                    # Determine line color based on occlusion status and track quality
                    track_quality = self.track_qualities.get(track_id, {}).get('overall', 0.5)
                    
                    if occlusion_status == 'fully_occluded':
                        base_color = (0, 0, 180)  # Red for fully occluded
                    elif occlusion_status == 'partially_occluded':
                        base_color = (0, 180, 180)  # Yellow for partially occluded
                    else:
                        base_color = (180, 0, 0)  # Blue for out of frame
                    
                    # Draw prediction lines
                    last_bbox = person_data.get('bbox')
                    if last_bbox is None:
                        continue
                        
                    # Start from center of current bbox
                    start_x = int((last_bbox[0] + last_bbox[2]) / 2)
                    start_y = int((last_bbox[1] + last_bbox[3]) / 2)
                    
                    # Draw each point in prediction
                    for i, (pred_x, pred_y, _, _) in enumerate(future_points):
                        # Skip if point is outside image bounds
                        if (pred_x < 0 or pred_x >= image.shape[1] or
                            pred_y < 0 or pred_y >= image.shape[0]):
                            continue
                        
                        # Calculate alpha (transparency) based on prediction distance
                        alpha = 1.0 - i / len(future_points)
                        
                        # Calculate point color with alpha
                        point_color = tuple(int(c * alpha) for c in base_color)
                        
                        # Draw line segment
                        if i == 0:
                            cv2.line(image, (start_x, start_y), (int(pred_x), int(pred_y)), point_color, 2)
                        else:
                            prev_x, prev_y = future_points[i-1][0], future_points[i-1][1]
                            cv2.line(image, (int(prev_x), int(prev_y)), (int(pred_x), int(pred_y)), point_color, 2)
                        
                        # Draw points with decreasing size
                        point_size = max(1, int(5 * (1.0 - i / len(future_points))))
                        cv2.circle(image, (int(pred_x), int(pred_y)), point_size, point_color, -1)
                        
                    # Add confidence indicator
                    if track_quality > 0.7:
                        # Draw a small indicator for high-quality prediction
                        label = f"{method.upper()} ({track_quality:.1f})"
                        cv2.putText(
                            image,
                            label,
                            (start_x + 5, start_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1
                        )
        except Exception as e:
            print(f"Error drawing predicted trajectories: {e}")
    
    def _draw_trajectories(self, image, tracked_persons):
        """
        Draw the historical trajectory paths for tracked persons.
        
        Args:
            image (numpy.ndarray): Image to draw on
            tracked_persons (dict): Dictionary of tracked persons
        """
        try:
            # Draw historical trajectory for each person
            for track_id, person_data in tracked_persons.items():
                # Skip if person has no valid bounding box
                if 'bbox' not in person_data or person_data['bbox'] is None:
                    continue
                
                # Get the track history for this person
                if track_id not in self.track_history:
                    continue
                
                history = self.track_history[track_id]
                if not history or len(history) < 2:
                    continue
                
                # Determine color based on the person ID or track ID
                person_id = person_data.get('person_id')
                if person_id is not None:
                    # Use a fixed color based on person ID for consistency
                    color_id = person_id % 255
                    color = (color_id, 255 - color_id, 128)
                else:
                    # Use a different color scheme for unidentified tracks
                    color_id = track_id % 255
                    color = (100, color_id, 255 - color_id)
                
                # Draw the trajectory lines connecting historical positions
                for i in range(1, len(history)):
                    prev_point = history[i-1]
                    curr_point = history[i]
                    
                    # Extract points (centers of bounding boxes)
                    prev_x, prev_y = int(prev_point[0]), int(prev_point[1])
                    curr_x, curr_y = int(curr_point[0]), int(curr_point[1])
                    
                    # Check if points are within image bounds
                    h, w = image.shape[:2]
                    if (0 <= prev_x < w and 0 <= prev_y < h and 
                        0 <= curr_x < w and 0 <= curr_y < h):
                        
                        # Calculate alpha (transparency) based on recency
                        frames_since = self.frame_count - curr_point[2]
                        alpha = max(0.3, 1.0 - (frames_since / 30.0))  # Fade older points
                        
                        # Apply alpha to color
                        alpha_color = tuple(int(c * alpha) for c in color)
                        
                        # Draw line segment with decreasing thickness based on age
                        thickness = max(1, int(3 - (i / len(history)) * 2))
                        cv2.line(image, (prev_x, prev_y), (curr_x, curr_y), alpha_color, thickness)
                
                # Draw points at each position with decreasing size for older points
                for i, point in enumerate(history):
                    x, y = int(point[0]), int(point[1])
                    
                    # Check if point is within image bounds
                    h, w = image.shape[:2]
                    if 0 <= x < w and 0 <= y < h:
                        # Calculate alpha (transparency) based on recency
                        frames_since = self.frame_count - point[2]
                        alpha = max(0.3, 1.0 - (frames_since / 30.0))
                        
                        # Apply alpha to color
                        alpha_color = tuple(int(c * alpha) for c in color)
                        
                        # Size decreases for older points
                        point_size = max(1, int(4 * (1.0 - i / len(history))))
                        cv2.circle(image, (x, y), point_size, alpha_color, -1)
        except Exception as e:
            print(f"Error in _draw_trajectories: {e}")

    def _analyze_occlusions(self, tracked_persons):
        """
        Analyze occlusions between persons to build an occlusion graph.
        This helps track which person is occluding whom and enables smarter occlusion handling.
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons
        """
        # ... existing code ...