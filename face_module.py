import os
import cv2
import numpy as np
import insightface
import torch
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


class FaceModule:
    def __init__(self, config, database=None):
        """
        Initialize the face detection and recognition module.
        
        Args:
            config (dict): Configuration parameters for face recognition
            database (DatabaseModule, optional): Database module for storing face embeddings
        """
        self.config = config
        self.detection_size = eval(config.get('detection_size', '(800, 800)'))
        self.recognition_threshold = config.get('recognition_threshold', 0.6)
        self.top_k = config.get('top_k', 1)
        
        # Database module for persistent storage
        self.database = database
        self.frame_count = 0
        self.db_update_period = 5  # Update database every N frames
        
        # Check for GPU availability
        self.use_gpu = config.get('use_gpu', True)
        self.ctx_id = -1  # Default to CPU
        
        try:
            if self.use_gpu and torch.cuda.is_available():
                print("Using GPU acceleration for face recognition")
                self.ctx_id = 0  # Use GPU
            else:
                print("GPU acceleration not available for face recognition, using CPU")
                self.ctx_id = -1  # Use CPU
        except ImportError:
            print("onnxruntime not available, using CPU for face recognition")
            self.ctx_id = -1
        
        # Initialize InsightFace model
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",  # Using a lightweight model
            root="./models",
            providers=['CUDAExecutionProvider'],
            allowed_modules=['detection', 'recognition'],
            det_size=(0, 0)    # Model will be downloaded to this directory
        )
        self.face_analyzer.prepare(ctx_id=self.ctx_id, det_size=self.detection_size)
        
        # Storage for recognized faces
        self.known_faces = {}
        self.current_tracked_faces = {}
        
    def detect_faces(self, frame):
        """
        Detect faces in the given frame.
        
        Args:
            frame (numpy.ndarray): Input image frame
            
        Returns:
            list: List of detected face objects
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Get face detection results
            faces = self.face_analyzer.get(frame)
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def track_faces(self, frame):
        """
        Track faces across frames and assign IDs.
        
        Args:
            frame (numpy.ndarray): Input image frame
            
        Returns:
            dict: Dictionary with face IDs as keys and face data as values
        """
        self.frame_count += 1
        
        faces = self.detect_faces(frame)
        
        # If no faces detected, update tracking data but don't return empty faces
        if not faces:
            # Keep existing tracked faces, but mark them as not seen in this frame
            for face_id, tracked_face in self.current_tracked_faces.items():
                if 'last_seen' not in tracked_face:
                    tracked_face['last_seen'] = self.frame_count - 1
                tracked_face['frames_since_seen'] = self.frame_count - tracked_face['last_seen']
            
            # Remove faces that haven't been seen for too long (120 frames = ~4 seconds at 30fps)
            faces_to_remove = [face_id for face_id, data in self.current_tracked_faces.items() 
                               if data.get('frames_since_seen', 0) > 120]
            for face_id in faces_to_remove:
                if face_id in self.current_tracked_faces:
                    del self.current_tracked_faces[face_id]
                    
            # We'll return faces that have been seen recently (within 15 frames = 0.5 seconds at 30fps)
            # This allows for brief occlusions without disappearing immediately
            recent_faces = {face_id: data for face_id, data in self.current_tracked_faces.items() 
                           if data.get('frames_since_seen', 0) <= 15}
            
            return recent_faces
        
        new_tracked_faces = {}
        
        # Create a mapping of person IDs to their initial face IDs
        # This helps us maintain consistency when multiple faces are detected for the same person
        person_to_initial_face = {}
        for face_id, tracked_face in self.current_tracked_faces.items():
            person_id = tracked_face.get('person_id', None)
            if person_id is not None:
                if person_id in person_to_initial_face:
                    # Keep track of the earliest face ID for each person
                    if face_id < person_to_initial_face[person_id]:
                        person_to_initial_face[person_id] = face_id
                else:
                    person_to_initial_face[person_id] = face_id
        
        # For each detected face, try to match with existing tracked faces
        for face in faces:
            face_embedding = face.embedding
            matched = False
            face_person_id = None
            
            # Try to match with existing faces
            if self.current_tracked_faces:
                for face_id, tracked_face in self.current_tracked_faces.items():
                    # Calculate cosine similarity
                    similarity = self._calculate_similarity(face_embedding, tracked_face['embedding'])
                    
                    if similarity > self.recognition_threshold:
                        # Update position and embedding with moving average
                        alpha = 0.3  # Weight for new observation
                        tracked_face['embedding'] = (1 - alpha) * tracked_face['embedding'] + alpha * face_embedding
                        tracked_face['bbox'] = face.bbox
                        tracked_face['kps'] = face.kps
                        tracked_face['det_score'] = face.det_score
                        tracked_face['last_seen'] = self.frame_count
                        tracked_face['frames_since_seen'] = 0
                        tracked_face['visible'] = True  # Mark as currently visible
                        
                        # Keep track of person ID if available
                        if 'person_id' in tracked_face:
                            face_person_id = tracked_face['person_id']
                        
                        # Update the database with this face detection
                        if self.database is not None and self.frame_count % self.db_update_period == 0:
                            # Update the person ID from the database, store this face embedding with the person
                            updated_person_id = self.database.update_person_for_face_embedding(
                                face_embedding, face_id, force_update=(face_person_id is None)
                            )
                            
                            if updated_person_id is not None:
                                face_person_id = updated_person_id
                                tracked_face['person_id'] = face_person_id
                                
                                # Add to our mapping of person IDs to initial face IDs
                                if face_person_id not in person_to_initial_face:
                                    person_to_initial_face[face_person_id] = face_id
                                elif face_id < person_to_initial_face[face_person_id]:
                                    person_to_initial_face[face_person_id] = face_id
                            
                        new_tracked_faces[face_id] = tracked_face
                        matched = True
                        break
            
            # If no match found, create new tracked face
            if not matched:
                # Initialize face_id as sequential
                new_id = len(self.current_tracked_faces) + 1 if not self.current_tracked_faces else max(self.current_tracked_faces.keys()) + 1
                
                # If we have a database, try to match with stored embeddings and get person_id
                if self.database is not None:
                    # This will either find a matching person or create a new one with the face ID
                    face_person_id = self.database.update_person_for_face_embedding(
                        face_embedding, new_id
                    )
                    
                    # If we got a valid person_id, check if this person already has a face ID
                    if face_person_id is not None:
                        # Check if this person ID already has an initial face ID
                        if face_person_id in person_to_initial_face:
                            # Use the existing face ID for consistency if that face isn't currently present
                            initial_face_id = person_to_initial_face[face_person_id]
                            if initial_face_id not in new_tracked_faces:
                                new_id = initial_face_id
                        else:
                            # Otherwise, use person_id as face_id for new detections for consistency
                            new_id = face_person_id
                            person_to_initial_face[face_person_id] = new_id
                
                new_tracked_faces[new_id] = {
                    'embedding': face_embedding,
                    'bbox': face.bbox,
                    'kps': face.kps,
                    'det_score': face.det_score,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'frames_since_seen': 0,
                    'visible': True,  # Mark as currently visible
                    'person_id': face_person_id  # May be None if no match in database
                }
                
                # Add to our mapping of person IDs to initial face IDs
                if face_person_id is not None and face_person_id not in person_to_initial_face:
                    person_to_initial_face[face_person_id] = new_id
        
        # Mark faces that weren't matched as not seen in this frame
        for face_id, tracked_face in self.current_tracked_faces.items():
            if face_id not in new_tracked_faces:
                if 'last_seen' not in tracked_face:
                    tracked_face['last_seen'] = self.frame_count - 1
                tracked_face['frames_since_seen'] = self.frame_count - tracked_face['last_seen']
                tracked_face['visible'] = False  # Mark as not currently visible
                
                # Keep faces that haven't been seen for a short time (15 frames = ~0.5 second at 30fps)
                # This is reduced from 30 to make faces disappear more quickly
                if tracked_face.get('frames_since_seen', 0) <= 15:
                    new_tracked_faces[face_id] = tracked_face
        
        # Update the current tracked faces
        self.current_tracked_faces = new_tracked_faces
        
        # Return only visible faces or those that were visible very recently (within 3 frames)
        visible_faces = {face_id: data for face_id, data in self.current_tracked_faces.items() 
                        if data.get('visible', False) or data.get('frames_since_seen', 0) <= 3}
        
        return visible_faces
    
    def draw_faces(self, frame, tracked_faces=None):
        """
        Draw bounding boxes and IDs on the faces.
        
        Args:
            frame (numpy.ndarray): Input image frame
            tracked_faces (dict, optional): Dictionary of tracked faces. If None, uses the current tracked faces.
            
        Returns:
            numpy.ndarray: Frame with drawn face information
        """
        if tracked_faces is None:
            # When using current_tracked_faces directly, only use visible faces
            tracked_faces = {face_id: data for face_id, data in self.current_tracked_faces.items() 
                            if data.get('visible', False) or data.get('frames_since_seen', 0) <= 3}
            
        if not tracked_faces:
            return frame
            
        result_frame = frame.copy()
        
        # Create a mapping of person IDs to the earliest assigned face ID
        person_to_initial_face = {}
        for face_id, face_data in tracked_faces.items():
            person_id = face_data.get('person_id', None)
            if person_id is not None:
                # If this person ID is already in our mapping, keep the smaller (earlier) face_id
                if person_id in person_to_initial_face:
                    if face_id < person_to_initial_face[person_id]:
                        person_to_initial_face[person_id] = face_id
                else:
                    person_to_initial_face[person_id] = face_id
        
        for face_id, face_data in tracked_faces.items():
            # Skip faces that are not visible currently or very recently
            if not face_data.get('visible', False) and face_data.get('frames_since_seen', 0) > 3:
                continue
                
            # Convert bbox to numpy array if it's a list
            if isinstance(face_data['bbox'], list):
                bbox = np.array(face_data['bbox'], dtype=np.int32)
            else:
                bbox = face_data['bbox'].astype(np.int32)
                
            # Validate bbox is within frame boundaries
            h, w = frame.shape[:2]
            bbox[0] = max(0, min(bbox[0], w-1))
            bbox[1] = max(0, min(bbox[1], h-1))
            bbox[2] = max(0, min(bbox[2], w-1))
            bbox[3] = max(0, min(bbox[3], h-1))
            
            # Skip invalid bbox
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            
            # Get person ID if available
            person_id = face_data.get('person_id', None)
            
            # Draw bounding box - green for recognized person, red for new face
            # Make recently disappeared faces semi-transparent
            frames_since_seen = face_data.get('frames_since_seen', 0)
            alpha = max(0.3, 1.0 - (frames_since_seen / 5.0)) if frames_since_seen > 0 else 1.0
            
            if person_id is not None:
                color = (0, int(255 * alpha), 0)  # Green with alpha
            else:
                color = (0, 0, int(255 * alpha))  # Red with alpha
            
            cv2.rectangle(
                result_frame, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                color, 
                2
            )
            
            # For consistent display: if this face belongs to a person with multiple faces,
            # use the initial face ID assigned to that person
            display_id = face_id
            if person_id is not None and person_id in person_to_initial_face:
                display_id = person_to_initial_face[person_id]
                
            # Display the appropriate label
            if person_id is not None:
                # For recognized persons, show both person ID and the initial face ID
                label = f"Person: {person_id} (Face: {display_id})"
            else:
                label = f"Face: {face_id}"
                
            cv2.putText(
                result_frame, 
                label, 
                (bbox[0], bbox[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )
            
            # Draw facial keypoints if needed
            if 'kps' in face_data:
                # Convert kps to numpy array if it's a list
                if isinstance(face_data['kps'], list):
                    kps = np.array(face_data['kps'], dtype=np.int32)
                else:
                    kps = face_data['kps'].astype(np.int32)
                    
                for i in range(kps.shape[0]):
                    cv2.circle(result_frame, (kps[i][0], kps[i][1]), 2, (0, 0, 255), -1)
                    
        return result_frame
    
    def get_person_id_from_face_id(self, face_id):
        """
        Get the person ID associated with a face ID.
        
        Args:
            face_id (int): The face ID
            
        Returns:
            int: Person ID if available, None otherwise
        """
        if face_id in self.current_tracked_faces:
            return self.current_tracked_faces[face_id].get('person_id', None)
        return None
    
    def update_face_person_id(self, face_id, person_id):
        """
        Update the person ID for a tracked face.
        
        Args:
            face_id (int): The face ID
            person_id (int): The person ID
        """
        if face_id in self.current_tracked_faces:
            self.current_tracked_faces[face_id]['person_id'] = person_id
    
    def _calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two face embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First face embedding
            embedding2 (numpy.ndarray): Second face embedding
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Flatten embeddings if they're not already flat
            embedding1 = embedding1.flatten()
            embedding2 = embedding2.flatten()
            
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 > 0:
                embedding1 = embedding1 / norm1
            if norm2 > 0:
                embedding2 = embedding2 / norm2
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Ensure the result is within valid range [-1, 1]
            similarity = max(-1.0, min(1.0, similarity))
            
            # Convert to positive similarity score [0, 1]
            return (similarity + 1) / 2 if similarity < 0 else similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0 