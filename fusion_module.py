import numpy as np
import cv2
import time
from collections import defaultdict

class FusionModule:
    """
    Module for fusing multiple biometric signals for more robust tracking
    """
    def __init__(self, config=None):
        """
        Initialize the fusion module
        
        Args:
            config (dict): Configuration parameters for fusion
        """
        self.config = config or {}
        
        # Weight parameters for different modalities in fusion
        self.weights = {
            'face': self.config.get('face_weight', 0.6),
            'gait': self.config.get('gait_weight', 0.25),
            'pose': self.config.get('pose_weight', 0.15),
            'appearance': self.config.get('appearance_weight', 0.2),
            'motion': self.config.get('motion_weight', 0.1),
        }
        
        # Thresholds for matching
        self.thresholds = {
            'face': self.config.get('face_threshold', 0.5),
            'gait': self.config.get('gait_threshold', 0.45),
            'pose': self.config.get('pose_threshold', 0.4),
            'appearance': self.config.get('appearance_threshold', 0.5),
            'fusion': self.config.get('fusion_threshold', 0.55),
        }
        
        # Store history of identifications for each track
        self.track_history = {}  # track_id -> {person_ids, timestamps, methods}
        
        # Keep confidence scores for each identification
        self.confidence_scores = {}  # track_id -> person_id -> {score, timestamp}
        
        # Store quality assessment for each modality
        self.quality_assessment = {}  # track_id -> modality -> quality_score
        
        # Maximum history length
        self.max_history = self.config.get('max_history', 30)
        
        # Adaptively adjust weights based on real-time quality assessment
        self.adaptive_weights = self.config.get('adaptive_weights', True)
        
        # Time window for temporal fusion (seconds)
        self.time_window = self.config.get('time_window', 5.0)
        
        # Track ID assignment history
        self.id_assignment_history = defaultdict(list)  # track_id -> list of assigned person_ids
        
    def update_quality_assessment(self, track_id, modality, quality_score):
        """
        Update quality assessment for a specific modality
        
        Args:
            track_id (int): Tracking ID
            modality (str): Modality name ('face', 'gait', 'appearance', etc.)
            quality_score (float): Quality score between 0-1
        """
        if track_id not in self.quality_assessment:
            self.quality_assessment[track_id] = {}
            
        self.quality_assessment[track_id][modality] = quality_score
        
        # Recalculate adaptive weights if needed
        if self.adaptive_weights:
            self._update_adaptive_weights(track_id)
    
    def _update_adaptive_weights(self, track_id):
        """
        Update weights based on quality assessment
        
        Args:
            track_id (int): Tracking ID
        """
        if track_id not in self.quality_assessment:
            return
            
        # Get quality scores for this track
        quality = self.quality_assessment[track_id]
        
        # Skip if not enough modalities assessed
        if len(quality) < 2:
            return
            
        # Calculate sum for normalization
        total_quality = sum(quality.values())
        
        if total_quality > 0:
            # Update weights proportionally to quality
            for modality in quality:
                if modality in self.weights:
                    # Scale the base weight by quality
                    base_weight = self.weights[modality]
                    quality_factor = quality[modality] / total_quality
                    
                    # Adjust weight (blend between base weight and quality-based weight)
                    adjusted_weight = 0.5 * base_weight + 0.5 * quality_factor
                    
                    # Store as track-specific weight
                    if not hasattr(self, 'track_weights'):
                        self.track_weights = {}
                    
                    if track_id not in self.track_weights:
                        self.track_weights[track_id] = {}
                        
                    self.track_weights[track_id][modality] = adjusted_weight
    
    def register_identification(self, track_id, person_id, confidence, method, timestamp=None):
        """
        Register an identification from any method
        
        Args:
            track_id (int): Tracking ID
            person_id (int): Person ID
            confidence (float): Confidence score (0-1)
            method (str): Identification method ('face', 'gait', etc.)
            timestamp (float, optional): Timestamp, uses current time if None
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Initialize track history if needed
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'person_ids': [],
                'timestamps': [],
                'methods': [],
                'confidence': []
            }
            
        # Add to history
        self.track_history[track_id]['person_ids'].append(person_id)
        self.track_history[track_id]['timestamps'].append(timestamp)
        self.track_history[track_id]['methods'].append(method)
        self.track_history[track_id]['confidence'].append(confidence)
        
        # Trim history if too long
        if len(self.track_history[track_id]['person_ids']) > self.max_history:
            self.track_history[track_id]['person_ids'] = self.track_history[track_id]['person_ids'][-self.max_history:]
            self.track_history[track_id]['timestamps'] = self.track_history[track_id]['timestamps'][-self.max_history:]
            self.track_history[track_id]['methods'] = self.track_history[track_id]['methods'][-self.max_history:]
            self.track_history[track_id]['confidence'] = self.track_history[track_id]['confidence'][-self.max_history:]
            
        # Update confidence scores
        if track_id not in self.confidence_scores:
            self.confidence_scores[track_id] = {}
            
        if person_id not in self.confidence_scores[track_id]:
            self.confidence_scores[track_id][person_id] = {
                'score': confidence,
                'timestamp': timestamp,
                'method': method,
                'history': [(confidence, timestamp, method)]
            }
        else:
            # Update with exponential moving average if same method
            current = self.confidence_scores[track_id][person_id]
            
            # Add to history
            current['history'].append((confidence, timestamp, method))
            
            # Keep history limited
            if len(current['history']) > self.max_history:
                current['history'] = current['history'][-self.max_history:]
            
            # Update based on method
            if method == current['method']:
                # Exponential moving average for same method
                alpha = 0.3  # Weight for new observation
                current['score'] = (1 - alpha) * current['score'] + alpha * confidence
            else:
                # For different method, use fusion
                current['score'] = self._fuse_confidence_scores(
                    current['score'], 
                    confidence,
                    current['method'],
                    method
                )
                
            current['timestamp'] = timestamp
            current['method'] = method
            
        # Record ID assignment
        self.id_assignment_history[track_id].append((person_id, confidence, method, timestamp))
    
    def _fuse_confidence_scores(self, score1, score2, method1, method2):
        """
        Fuse confidence scores from different methods
        
        Args:
            score1 (float): First confidence score
            score2 (float): Second confidence score
            method1 (str): First method
            method2 (str): Second method
            
        Returns:
            float: Fused confidence score
        """
        # Get weights for each method
        weight1 = self.weights.get(method1, 0.5)
        weight2 = self.weights.get(method2, 0.5)
        
        # Normalize weights
        total_weight = weight1 + weight2
        weight1 /= total_weight
        weight2 /= total_weight
        
        # Weighted average
        return weight1 * score1 + weight2 * score2
    
    def get_fused_person_id(self, track_id, current_time=None):
        """
        Get fused person ID for a track based on history
        
        Args:
            track_id (int): Tracking ID
            current_time (float, optional): Current timestamp, uses time.time() if None
            
        Returns:
            tuple: (person_id, confidence, method) or (None, 0, None) if not found
        """
        if current_time is None:
            current_time = time.time()
            
        if track_id not in self.track_history:
            return None, 0, None
            
        # Get track history
        history = self.track_history[track_id]
        
        # Filter recent identifications within time window
        recent_indices = []
        for i, ts in enumerate(history['timestamps']):
            if current_time - ts <= self.time_window:
                recent_indices.append(i)
                
        if not recent_indices:
            return None, 0, None
            
        # Count person IDs in recent history
        person_counts = {}
        method_counts = {}
        weighted_scores = {}
        
        for i in recent_indices:
            person_id = history['person_ids'][i]
            method = history['methods'][i]
            confidence = history['confidence'][i]
            timestamp = history['timestamps'][i]
            
            # Calculate recency weight (more recent = higher weight)
            recency_weight = 1.0 - (current_time - timestamp) / self.time_window
            
            # Get method weight
            method_weight = self.weights.get(method, 0.5)
            
            # Calculate weighted score
            weighted_score = confidence * method_weight * recency_weight
            
            if person_id not in weighted_scores:
                weighted_scores[person_id] = []
                person_counts[person_id] = 0
                
            weighted_scores[person_id].append(weighted_score)
            person_counts[person_id] += 1
            
            if method not in method_counts:
                method_counts[method] = 0
            method_counts[method] += 1
            
        # Find person ID with highest weighted score
        best_person_id = None
        best_score = 0
        best_method = None
        
        for person_id, scores in weighted_scores.items():
            avg_score = sum(scores) / len(scores)
            count_weight = min(1.0, person_counts[person_id] / 5.0)  # Cap at 5 observations
            
            final_score = avg_score * count_weight
            
            if final_score > best_score:
                best_score = final_score
                best_person_id = person_id
                
        # Determine best method (most frequent for this ID)
        if best_person_id is not None:
            method_for_best_id = {}
            for i in recent_indices:
                if history['person_ids'][i] == best_person_id:
                    method = history['methods'][i]
                    if method not in method_for_best_id:
                        method_for_best_id[method] = 0
                    method_for_best_id[method] += 1
                    
            if method_for_best_id:
                best_method = max(method_for_best_id.items(), key=lambda x: x[1])[0]
            
        return best_person_id, best_score, best_method
    
    def process_tracked_persons(self, tracked_persons):
        """
        Process tracked persons and update with fused identities
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons
            
        Returns:
            dict: Updated tracked persons with fused identities
        """
        current_time = time.time()
        
        for track_id, person in tracked_persons.items():
            # Skip if already identified in this frame
            if 'fusion_processed' in person and person['fusion_processed']:
                continue
                
            # Process identifications from different modalities
            self._process_identifications(track_id, person, current_time)
            
            # Get fused identity
            person_id, confidence, method = self.get_fused_person_id(track_id, current_time)
            
            # Update person data if confidence meets threshold
            if person_id is not None and confidence >= self.thresholds['fusion']:
                person['person_id'] = person_id
                person['id_confidence'] = confidence
                person['id_method'] = f"fusion:{method}"
                
            # Mark as processed
            person['fusion_processed'] = True
            
        return tracked_persons
    
    def _process_identifications(self, track_id, person, current_time):
        """
        Process identifications from different modalities
        
        Args:
            track_id (int): Tracking ID
            person (dict): Person data
            current_time (float): Current timestamp
        """
        # Process face identification
        if 'face_id' in person and person.get('person_id') is not None:
            # Face recognition confidence (if available)
            confidence = person.get('face_confidence', 0.8)
            
            # Register identification
            self.register_identification(
                track_id, 
                person['person_id'],
                confidence,
                'face',
                current_time
            )
            
            # Update quality assessment based on face detection score if available
            if 'face_det_score' in person:
                self.update_quality_assessment(track_id, 'face', person['face_det_score'])
                
        # Process gait identification
        if 'gait_signature' in person and person.get('gait_person_id') is not None:
            confidence = person.get('gait_confidence', 0.7)
            
            # Register identification
            self.register_identification(
                track_id,
                person['gait_person_id'],
                confidence,
                'gait',
                current_time
            )
            
            # Update quality assessment - use movement speed as a proxy for gait quality
            if 'velocity' in person:
                # Motion quality increases with speed up to a point (walking speed)
                speed = np.linalg.norm(person['velocity'])
                motion_quality = min(1.0, speed / 3.0)  # Normalize to [0,1], assumes 3 is typical walking speed
                self.update_quality_assessment(track_id, 'gait', motion_quality)
                
        # Process appearance-based identification
        if 'appearance' in person and person.get('appearance_person_id') is not None:
            confidence = person.get('appearance_confidence', 0.6)
            
            # Register identification
            self.register_identification(
                track_id,
                person['appearance_person_id'],
                confidence,
                'appearance',
                current_time
            )
            
            # Update quality based on lighting conditions if available
            if 'appearance_quality' in person:
                self.update_quality_assessment(track_id, 'appearance', person['appearance_quality'])
                
        # Process pose-based identification
        if 'keypoints' in person and person.get('pose_person_id') is not None:
            confidence = person.get('pose_confidence', 0.5)
            
            # Register identification
            self.register_identification(
                track_id,
                person['pose_person_id'],
                confidence,
                'pose',
                current_time
            )
            
            # Update quality based on number of detected keypoints
            keypoints = person['keypoints']
            valid_keypoints = sum(1 for kp in keypoints if kp is not None)
            pose_quality = valid_keypoints / len(keypoints)
            self.update_quality_assessment(track_id, 'pose', pose_quality)
    
    def get_identification_stability(self, track_id, time_window=None):
        """
        Calculate stability of identification for a track
        
        Args:
            track_id (int): Tracking ID
            time_window (float, optional): Time window, uses default if None
            
        Returns:
            float: Stability score between 0-1
        """
        if time_window is None:
            time_window = self.time_window
            
        if track_id not in self.id_assignment_history or len(self.id_assignment_history[track_id]) < 2:
            return 1.0  # Not enough history
            
        # Get history within time window
        current_time = time.time()
        recent_history = [
            (pid, conf, method, ts) for pid, conf, method, ts in self.id_assignment_history[track_id]
            if current_time - ts <= time_window
        ]
        
        if not recent_history:
            return 1.0
            
        # Count unique IDs and their frequency
        id_counts = {}
        for pid, _, _, _ in recent_history:
            if pid not in id_counts:
                id_counts[pid] = 0
            id_counts[pid] += 1
            
        # Calculate stability as ratio of most frequent ID to total
        total_assignments = len(recent_history)
        most_frequent_count = max(id_counts.values())
        
        stability = most_frequent_count / total_assignments
        return stability
    
    def is_id_switch_suspected(self, track_id, person_id):
        """
        Check if an ID switch is suspected for this track
        
        Args:
            track_id (int): Tracking ID
            person_id (int): Current person ID
            
        Returns:
            bool: True if ID switch is suspected
        """
        if track_id not in self.id_assignment_history or len(self.id_assignment_history[track_id]) < 3:
            return False
            
        # Get recent history
        recent_history = self.id_assignment_history[track_id][-5:]  # Last 5 assignments
        
        # Count unique IDs
        id_counts = {}
        for pid, _, _, _ in recent_history:
            if pid not in id_counts:
                id_counts[pid] = 0
            id_counts[pid] += 1
            
        # If current ID is not the most frequent or appeared suddenly, suspect switch
        if len(id_counts) > 1:
            most_frequent_id = max(id_counts.items(), key=lambda x: x[1])[0]
            
            if person_id != most_frequent_id:
                # Current ID is different from the most frequent
                return True
                
            # Check for sudden appearance of this ID after consistent different ID
            if len(recent_history) >= 3:
                prev_ids = [pid for pid, _, _, _ in recent_history[:-1]]
                if person_id not in prev_ids and len(set(prev_ids)) == 1:
                    return True
                    
        return False 