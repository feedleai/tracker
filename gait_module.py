import numpy as np
import cv2
import os
import time
from pathlib import Path
from scipy.signal import savgol_filter
import pickle
from sklearn.preprocessing import normalize

class GaitModule:
    """
    Module for gait recognition and identification
    """
    def __init__(self, config=None):
        """
        Initialize the gait recognition module
        
        Args:
            config (dict): Configuration parameters for gait recognition
        """
        self.config = config or {}
        
        # Extraction parameters
        self.sequence_length = self.config.get('sequence_length', 20)  # Number of frames to use for gait analysis
        self.min_track_length = self.config.get('min_track_length', 10)  # Minimum track length to extract gait
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)  # Threshold for gait matching
        self.smoothing_window = self.config.get('smoothing_window', 7)  # Window size for motion smoothing
        self.feature_dim = self.config.get('feature_dim', 128)  # Dimension of extracted gait features
        
        # Storage for gait signatures
        self.gait_database = {}  # person_id -> gait_signature
        self.pending_signatures = {}  # track_id -> list of motion data
        
        # Database path for persistence
        self.db_path = self.config.get('db_path', 'gait_database.pkl')
        
        # Load existing database if available
        self._load_database()
    
    def _load_database(self):
        """Load gait database from disk if available"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.gait_database = pickle.load(f)
                print(f"Loaded gait database with {len(self.gait_database)} entries")
            except Exception as e:
                print(f"Error loading gait database: {e}")
                self.gait_database = {}
    
    def _save_database(self):
        """Save gait database to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.gait_database, f)
        except Exception as e:
            print(f"Error saving gait database: {e}")
    
    def update_motion_sequence(self, track_id, person_data):
        """
        Update motion sequence for a tracked person
        
        Args:
            track_id (int): Tracking ID of the person
            person_data (dict): Person tracking data including bounding box
        """
        if 'bbox' not in person_data or 'velocity' not in person_data:
            return
        
        # Extract motion features
        bbox = person_data['bbox']
        velocity = person_data['velocity'] if 'velocity' in person_data else np.zeros(2)
        
        # Calculate positional and movement features
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        aspect_ratio = width / height if height > 0 else 0
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Create motion feature vector
        motion_feature = np.array([
            center_x, center_y,
            width, height, 
            aspect_ratio,
            velocity[0], velocity[1],
            # Add more features like joint positions if available
        ])
        
        # Initialize or update the pending signature
        if track_id not in self.pending_signatures:
            self.pending_signatures[track_id] = []
        
        # Add motion feature to the sequence
        self.pending_signatures[track_id].append(motion_feature)
        
        # Keep only the most recent frames
        if len(self.pending_signatures[track_id]) > self.sequence_length:
            self.pending_signatures[track_id] = self.pending_signatures[track_id][-self.sequence_length:]
    
    def extract_gait_signature(self, track_id):
        """
        Extract gait signature from collected motion sequence
        
        Args:
            track_id (int): Tracking ID of the person
            
        Returns:
            numpy.ndarray or None: Extracted gait signature or None if not enough data
        """
        if track_id not in self.pending_signatures:
            return None
        
        motion_sequence = self.pending_signatures[track_id]
        
        # Check if we have enough frames
        if len(motion_sequence) < self.min_track_length:
            return None
        
        # Convert to numpy array
        motion_array = np.array(motion_sequence)
        
        # Apply smoothing to reduce noise
        smoothed_array = np.zeros_like(motion_array)
        for i in range(motion_array.shape[1]):
            # Apply Savitzky-Golay filter for smoothing if we have enough data points
            if len(motion_array) >= self.smoothing_window:
                # Make sure window_length is odd and less than sequence length
                window_length = min(self.smoothing_window, len(motion_array))
                window_length = window_length if window_length % 2 == 1 else window_length - 1
                
                if window_length >= 3:  # Minimum window size for savgol_filter
                    smoothed_array[:, i] = savgol_filter(
                        motion_array[:, i], 
                        window_length=window_length, 
                        polyorder=2
                    )
                else:
                    smoothed_array[:, i] = motion_array[:, i]
            else:
                smoothed_array[:, i] = motion_array[:, i]
        
        # Extract various gait features
        # 1. Calculate derivatives (velocity and acceleration patterns)
        velocity = np.diff(smoothed_array, axis=0)
        
        # If we have enough data for acceleration
        if velocity.shape[0] > 1:
            acceleration = np.diff(velocity, axis=0)
            
            # Compute statistical features
            mean_pos = np.mean(smoothed_array, axis=0)
            std_pos = np.std(smoothed_array, axis=0)
            mean_vel = np.mean(velocity, axis=0)
            std_vel = np.std(velocity, axis=0)
            mean_acc = np.mean(acceleration, axis=0)
            std_acc = np.std(acceleration, axis=0)
            
            # Compute step frequency if possible (simplified)
            step_freq = 0
            if len(velocity) > 5:
                # Look at vertical velocity pattern
                y_velocity = velocity[:, 1]  # Assuming second column is y-velocity
                # Count zero crossings as a proxy for steps
                zero_crossings = np.where(np.diff(np.signbit(y_velocity)))[0]
                if len(zero_crossings) > 1:
                    step_freq = len(zero_crossings) / len(velocity)
            
            # Combine all features into a gait signature
            gait_signature = np.concatenate([
                mean_pos, std_pos,
                mean_vel, std_vel, 
                mean_acc, std_acc,
                [step_freq]
            ])
            
            # Normalize the signature
            if np.any(gait_signature):  # Check if not all zeros
                gait_signature = normalize(gait_signature.reshape(1, -1))[0]
                
            return gait_signature
            
        return None
    
    def compare_gait_signatures(self, signature1, signature2):
        """
        Compare two gait signatures for similarity
        
        Args:
            signature1 (numpy.ndarray): First gait signature
            signature2 (numpy.ndarray): Second gait signature
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if signature1 is None or signature2 is None:
            return 0
        
        # Make sure signatures have the same length
        min_len = min(len(signature1), len(signature2))
        sig1 = signature1[:min_len]
        sig2 = signature2[:min_len]
        
        # Compute cosine similarity
        dot_product = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        similarity = dot_product / (norm1 * norm2)
        return max(0, similarity)  # Ensure non-negative
    
    def find_matching_person(self, gait_signature, threshold=None):
        """
        Find a matching person ID for a given gait signature
        
        Args:
            gait_signature (numpy.ndarray): Gait signature to match
            threshold (float, optional): Similarity threshold, uses default if None
            
        Returns:
            int or None: Matching person ID or None if no match
        """
        if gait_signature is None or not self.gait_database:
            return None
        
        threshold = threshold or self.similarity_threshold
        best_match = None
        best_similarity = 0
        
        for person_id, stored_signature in self.gait_database.items():
            similarity = self.compare_gait_signatures(gait_signature, stored_signature)
            
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id
        
        return best_match
    
    def update_person_gait(self, person_id, gait_signature):
        """
        Update or add a person's gait signature in the database
        
        Args:
            person_id (int): Person ID
            gait_signature (numpy.ndarray): Gait signature
        """
        if gait_signature is None:
            return
        
        if person_id in self.gait_database:
            # Average with existing signature for more stable representation
            existing_sig = self.gait_database[person_id]
            updated_sig = (existing_sig + gait_signature) / 2
            # Normalize again
            if np.any(updated_sig):  # Check if not all zeros
                updated_sig = normalize(updated_sig.reshape(1, -1))[0]
            self.gait_database[person_id] = updated_sig
        else:
            # Add new entry
            self.gait_database[person_id] = gait_signature
        
        # Save database whenever it's updated
        self._save_database()
    
    def update_tracked_persons(self, tracked_persons, person_module):
        """
        Update gait signatures for all tracked persons and attempt re-identification
        
        Args:
            tracked_persons (dict): Dictionary of tracked persons
            person_module (object): Reference to person tracking module for ID updates
            
        Returns:
            dict: Updated tracked_persons with any gait-based identifications
        """
        for track_id, person in tracked_persons.items():
            # Update motion sequence for this person
            self.update_motion_sequence(track_id, person)
            
            # Get existing person ID if available
            existing_person_id = person.get('person_id')
            
            # If person already has an ID and we have enough motion data,
            # update their gait signature in the database
            if existing_person_id is not None:
                gait_signature = self.extract_gait_signature(track_id)
                if gait_signature is not None:
                    # Store gait feature in database
                    person['gait_signature'] = gait_signature
                    self.update_person_gait(existing_person_id, gait_signature)
            
            # If no person ID yet, try to identify by gait
            elif 'person_id' not in person or person['person_id'] is None:
                gait_signature = self.extract_gait_signature(track_id)
                
                if gait_signature is not None:
                    # Store gait feature
                    person['gait_signature'] = gait_signature
                    
                    # Try to find a match in the database
                    person_id = self.find_matching_person(gait_signature)
                    
                    if person_id is not None:
                        # If found, set person ID
                        person['person_id'] = person_id
                        person['id_method'] = 'gait'
                        
                        # Update person tracking module's internal mapping
                        if hasattr(person_module, 'tracker_to_person_map'):
                            person_module.tracker_to_person_map[track_id] = person_id
                        
                        # Update the gait database with new observation
                        self.update_person_gait(person_id, gait_signature)
                        
                        print(f"Identified person {person_id} by gait pattern (track ID: {track_id})")
        
        return tracked_persons 