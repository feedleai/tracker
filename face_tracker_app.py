import os
import cv2
import argparse
import time
import sys
from config_module import ConfigModule
from camera_module import CameraModule
from person_tracking_module import PersonTrackingModule

# Check if we need to use the InsightFace stub
try:
    from face_module import FaceModule
    USING_STUB = False
except ImportError:
    print("Cannot import FaceModule, checking for stub...")
    # Try to import from stub
    if os.path.exists("insightface_stub"):
        sys.path.insert(0, os.path.abspath("insightface_stub"))
        try:
            from face_module import FaceModule
            USING_STUB = True
            print("Using InsightFace stub for face detection")
        except ImportError:
            print("ERROR: Cannot import FaceModule from stub either")
            USING_STUB = True
    else:
        print("ERROR: InsightFace not installed and stub not found")
        USING_STUB = True

# Only import database if not using stub
if not USING_STUB:
    try:
        from database_module import DatabaseModule
    except ImportError:
        print("WARNING: DatabaseModule not available, face recognition persistence disabled")
        DatabaseModule = None
else:
    DatabaseModule = None


class FaceTrackerApp:
    def __init__(self, config_path="config.yml"):
        """
        Initialize the face tracker application.
        
        Args:
            config_path (str): Path to the configuration file
        """
        print("Initializing Face Tracker Application...")
        
        # Load configuration
        self.config_module = ConfigModule(config_path)
        
        # Initialize database module if enabled and available
        self.database = None
        if not USING_STUB and DatabaseModule is not None:
            db_config = self.config_module.get_database_config()
            if db_config.get('use_db', True):
                print("Initializing database...")
                db_path = db_config.get('db_path', 'face_database.db')
                self.database = DatabaseModule(db_path)
        
        # Initialize camera module
        camera_configs = self.config_module.get_cameras_config()
        self.camera_module = CameraModule(camera_configs)
        
        # Initialize face recognition module
        face_recognition_config = self.config_module.get_face_recognition_config()
        self.face_module = FaceModule(face_recognition_config, self.database)
        
        # Initialize person tracking module
        person_tracking_config = self.config_module.get_person_tracking_config()
        self.person_module = PersonTrackingModule(person_tracking_config)
        
        # Store historical mapping between person tracking IDs and face IDs
        # This helps maintain consistent face IDs when faces reappear
        self.person_to_face_history = {}
        
        # Get output configuration
        self.output_config = self.config_module.get_output_config()
        
        # Create output directory if needed
        if self.output_config.get('save_detections', False):
            output_dir = self.output_config.get('output_dir', './detected_faces')
            os.makedirs(output_dir, exist_ok=True)
            
        # Application state
        self.running = False
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        
        # Video output configuration
        self.video_writer = None
        self.output_video_path = "output.mp4"
        self.output_video_fps = 30
        self.output_video_size = None  # Will be set based on first frame
        
        # Check for GPU availability - combining information from both modules
        self.using_gpu_face = getattr(self.face_module, 'ctx_id', -1) >= 0
        self.using_gpu_person = getattr(self.person_module, 'device', 'cpu') != 'cpu'
        self.using_gpu = self.using_gpu_face or self.using_gpu_person
        
        if self.using_gpu:
            if self.using_gpu_face and self.using_gpu_person:
                print("GPU acceleration is available and will be used for both face recognition and person tracking")
            elif self.using_gpu_face:
                print("GPU acceleration is available and will be used for face recognition only")
            elif self.using_gpu_person:
                print("GPU acceleration is available and will be used for person tracking only")
        else:
            print("GPU acceleration is not available, using CPU")
        
    def start(self):
        """Start the face tracker application"""
        print("Starting Face Tracker Application...")
        
        # Start cameras
        started_cameras = self.camera_module.start_all_cameras()
        if not started_cameras:
            print("Failed to start any cameras. Please check your configuration.")
            return False
            
        print(f"Started cameras: {', '.join(started_cameras)}")
        print(f"Active camera: {self.camera_module.get_active_camera_name()}")
        
        self.running = True
        self.start_time = time.time()
        self.run_tracking_loop()
        
        return True
    
    def run_tracking_loop(self):
        """Run the main tracking loop"""
        print("Running tracking loop. Press 'q' to quit, 's' to switch cameras.")
        
        while self.running:
            # Get frame from active camera
            ret, frame = self.camera_module.get_camera_frame()
            
            if not ret or frame is None:
                print("Failed to get frame from camera")
                time.sleep(0.1)
                continue
                
            # Create working copy of the frame
            working_frame = frame.copy()
            result_frame = frame.copy()
            
            # Step 1: Detect and track persons using YOLO and BoTSORT
            tracked_persons = self.person_module.detect_and_track(working_frame)
            
            # Step 2: Detect and track faces
            tracked_faces = self.face_module.track_faces(working_frame)
            
            # Step 3: Associate tracked persons with face identities
            tracked_persons = self.person_module.update_face_associations(
                tracked_persons, 
                tracked_faces,
                self.face_module  # Pass face module to get person IDs
            )
            
            # Create a mapping of person IDs to face IDs for consistent identity assignment
            person_to_face_map = {}
            for face_id, face_data in tracked_faces.items():
                person_id = face_data.get('person_id', None)
                if person_id is not None:
                    if person_id not in person_to_face_map:
                        person_to_face_map[person_id] = []
                    person_to_face_map[person_id].append(face_id)
            
            # Update the person-to-face history mapping for consistent ID assignment
            for track_id, person in tracked_persons.items():
                if person.get('face_id') is not None:
                    self.person_to_face_history[track_id] = {
                        'face_id': person['face_id'],
                        'person_id': person.get('person_id', None),
                        'last_seen': self.frame_count,
                        'appearance': person.get('appearance', None)
                    }
            
            # Step 4: Get face crops from tracked persons if no face detected
            person_face_crops = self.person_module.get_person_face_crops(working_frame, tracked_persons)
            
            # Process face crops for persons without face IDs
            for track_id, crop_data in person_face_crops.items():
                person = tracked_persons.get(track_id)
                
                # Only process if this person doesn't have a face ID yet
                if person and person.get('face_id') is None:
                    # First, try to use appearance features for reidentification
                    # If this person has similar appearance to a previously known person, use that identity
                    self._try_reidentify_by_appearance(track_id, person, person_to_face_map)
                    
                    # If still no face ID, check if this person track_id has a historical face association
                    if person.get('face_id') is None and track_id in self.person_to_face_history:
                        history = self.person_to_face_history[track_id]
                        # Enhanced: Don't limit by frame count for historical matches
                        # Use history regardless of when last seen - this helps with long-term reidentification
                        # Restore the historical face ID and person ID
                        person['face_id'] = history['face_id']
                        self.person_module.tracker_to_face_map[track_id] = history['face_id']
                        
                        if history['person_id'] is not None:
                            person['person_id'] = history['person_id']
                            self.person_module.tracker_to_person_map[track_id] = history['person_id']
                            
                            # Update database with this association if we have a database
                            if self.database is not None and not USING_STUB:
                                self.database.map_face_to_person(
                                    history['face_id'], 
                                    history['person_id']
                                )
                        
                        # Update the last seen time
                        history['last_seen'] = self.frame_count
                        
                        # Skip further face detection for this person
                        continue
                    
                    # If still no face ID and we have a database, try to find a long-term match from the database
                    if person.get('face_id') is None and self.database is not None and not USING_STUB:
                        if 'appearance' in person and person['appearance'] is not None:
                            # First attempt database reidentification by appearance features
                            try:
                                # Get the crop data for this person
                                crop_data = self._get_person_crop(frame, person['bbox'])
                                if crop_data and 'crop' in crop_data:
                                    # Run face detection on the cropped region
                                    faces = self.face_module.detect_faces(crop_data['crop'])
                                    if faces and len(faces) > 0:
                                        # Get face embedding and try to match with database
                                        face_embedding = faces[0].embedding
                                        person_id = self.database.find_matching_person(face_embedding, threshold=0.5)
                                        
                                        if person_id is not None:
                                            # Found a database match - create a new face ID
                                            new_id = self.face_module._generate_unique_face_id()
                                            
                                            # Associate with this person ID
                                            person['face_id'] = new_id
                                            person['person_id'] = person_id
                                            self.person_module.tracker_to_face_map[track_id] = new_id
                                            self.person_module.tracker_to_person_map[track_id] = person_id
                                            
                                            # Add to tracked faces
                                            if hasattr(self.face_module, 'current_tracked_faces'):
                                                self.face_module.current_tracked_faces[new_id] = {
                                                    'embedding': face_embedding,
                                                    'bbox': crop_data['adjusted_bbox'],
                                                    'kps': faces[0].kps if hasattr(faces[0], 'kps') else None,
                                                    'det_score': faces[0].det_score if hasattr(faces[0], 'det_score') else 1.0,
                                                    'first_seen': getattr(self.face_module, 'frame_count', 0),
                                                    'last_seen': getattr(self.face_module, 'frame_count', 0),
                                                    'frames_since_seen': 0,
                                                    'visible': True,
                                                    'person_id': person_id
                                                }
                                            
                                            # Store in history
                                            self.person_to_face_history[track_id] = {
                                                'face_id': new_id,
                                                'person_id': person_id,
                                                'last_seen': self.frame_count,
                                                'appearance': person.get('appearance', None)
                                            }
                                            
                                            # Update database mapping
                                            self.database.map_face_to_person(new_id, person_id)
                                            
                                            # Skip further processing
                                            continue
                            except Exception as e:
                                print(f"Error in database reidentification: {e}")
                    
                    # Run face detection on the cropped region
                    faces = self.face_module.detect_faces(crop_data['crop'])
                    
                    # If a face is found, track it
                    if faces:
                        try:
                            # Get the face embedding from the detection
                            face_embedding = faces[0].embedding
                            
                            # Try to match with existing tracked faces or create a new tracked face
                            matched = False
                            
                            # Try to match with existing faces by embedding similarity
                            if hasattr(self.face_module, 'current_tracked_faces') and self.face_module.current_tracked_faces:
                                for face_id, tracked_face in self.face_module.current_tracked_faces.items():
                                    try:
                                        similarity = self.face_module._calculate_similarity(
                                            face_embedding, tracked_face['embedding']
                                        )
                                        
                                        if similarity > self.face_module.recognition_threshold:
                                            # Get person ID if available
                                            person_id = tracked_face.get('person_id', None)
                                            
                                            # Associate this person with this face ID and person ID
                                            person['face_id'] = face_id
                                            self.person_module.tracker_to_face_map[track_id] = face_id
                                            
                                            if person_id is not None:
                                                person['person_id'] = person_id
                                                self.person_module.tracker_to_person_map[track_id] = person_id
                                                
                                                # Update the database with this association
                                                if self.database is not None and not USING_STUB:
                                                    self.database.map_face_to_person(face_id, person_id)
                                            
                                            # Store in history for future frames
                                            self.person_to_face_history[track_id] = {
                                                'face_id': face_id,
                                                'person_id': person_id,
                                                'last_seen': self.frame_count,
                                                'appearance': person.get('appearance', None)
                                            }
                                                
                                            matched = True
                                            break
                                    except Exception as e:
                                        print(f"Error matching face: {e}")
                            
                            # If no match and database available, try to match with database
                            if not matched and self.database is not None and not USING_STUB:
                                try:
                                    # Use the enhanced database function to get or create the right person ID
                                    # This will create a new ID if needed or match with an existing person
                                    new_face_id = len(self.face_module.current_tracked_faces) + 1 if not self.face_module.current_tracked_faces else max(self.face_module.current_tracked_faces.keys()) + 1
                                    person_id = self.database.update_person_for_face_embedding(
                                        face_embedding,
                                        new_face_id
                                    )
                                    
                                    if person_id is not None:
                                        # Use the person_id directly as the face ID for consistency
                                        new_id = person_id
                                        
                                        # Adjust the face bbox to frame coordinates
                                        original_bbox = crop_data['bbox']
                                        face_bbox = faces[0].bbox
                                        
                                        # Convert face bbox from crop coordinates to frame coordinates
                                        adjusted_bbox = [
                                            original_bbox[0] + face_bbox[0],
                                            original_bbox[1] + face_bbox[1],
                                            original_bbox[0] + face_bbox[2],
                                            original_bbox[1] + face_bbox[3]
                                        ]
                                        
                                        # Add the new face to tracked faces
                                        if hasattr(self.face_module, 'current_tracked_faces'):
                                            self.face_module.current_tracked_faces[new_id] = {
                                                'embedding': face_embedding,
                                                'bbox': adjusted_bbox,
                                                'kps': faces[0].kps,
                                                'det_score': faces[0].det_score,
                                                'first_seen': getattr(self.face_module, 'frame_count', 0),
                                                'last_seen': getattr(self.face_module, 'frame_count', 0),
                                                'frames_since_seen': 0,
                                                'visible': True,
                                                'person_id': person_id
                                            }
                                        
                                        # Associate this person with this new face ID and person ID
                                        person['face_id'] = new_id
                                        person['person_id'] = person_id
                                        self.person_module.tracker_to_face_map[track_id] = new_id
                                        self.person_module.tracker_to_person_map[track_id] = person_id
                                        
                                        # Store in history for future frames
                                        self.person_to_face_history[track_id] = {
                                            'face_id': new_id,
                                            'person_id': person_id,
                                            'last_seen': self.frame_count,
                                            'appearance': person.get('appearance', None)
                                        }
                                        
                                        matched = True
                                except Exception as e:
                                    print(f"Error matching with database: {e}")
                            
                            # If still no match, create a new face ID and store in database if enabled
                            if not matched and hasattr(self.face_module, 'current_tracked_faces'):
                                try:
                                    # Create a new tracked face from this detection
                                    new_id = len(self.face_module.current_tracked_faces) + 1 if not self.face_module.current_tracked_faces else max(self.face_module.current_tracked_faces.keys()) + 1
                                    
                                    # Adjust the face bbox to frame coordinates
                                    original_bbox = crop_data['bbox']
                                    face_bbox = faces[0].bbox
                                    
                                    # Convert face bbox from crop coordinates to frame coordinates
                                    adjusted_bbox = [
                                        original_bbox[0] + face_bbox[0],
                                        original_bbox[1] + face_bbox[1],
                                        original_bbox[0] + face_bbox[2],
                                        original_bbox[1] + face_bbox[3]
                                    ]
                                    
                                    # Add to database if enabled and get a person ID
                                    person_id = None
                                    if self.database is not None and not USING_STUB:
                                        try:
                                            # Add person with the face ID as initial person ID for consistency
                                            person_id = self.database.update_person_for_face_embedding(
                                                face_embedding, 
                                                new_id
                                            )
                                            
                                            if person_id is not None:
                                                # Use the returned person ID as the face ID for consistency
                                                new_id = person_id
                                        except Exception as e:
                                            print(f"Error adding to database: {e}")
                                    
                                    # Add the new face to tracked faces
                                    self.face_module.current_tracked_faces[new_id] = {
                                        'embedding': face_embedding,
                                        'bbox': adjusted_bbox,
                                        'kps': faces[0].kps,
                                        'det_score': faces[0].det_score,
                                        'first_seen': getattr(self.face_module, 'frame_count', 0),
                                        'last_seen': getattr(self.face_module, 'frame_count', 0),
                                        'frames_since_seen': 0,
                                        'visible': True,
                                        'person_id': person_id
                                    }
                                    
                                    # Associate this person with this new face ID and person ID if available
                                    person['face_id'] = new_id
                                    self.person_module.tracker_to_face_map[track_id] = new_id
                                    
                                    if person_id is not None:
                                        person['person_id'] = person_id
                                        self.person_module.tracker_to_person_map[track_id] = person_id
                                    
                                    # Store in history for future frames
                                    self.person_to_face_history[track_id] = {
                                        'face_id': new_id,
                                        'person_id': person_id,
                                        'last_seen': self.frame_count,
                                        'appearance': person.get('appearance', None)
                                    }
                                except Exception as e:
                                    print(f"Error creating new face: {e}")
                        except Exception as e:
                            print(f"Error processing face crop: {e}")
            
            # Step 5: Draw visualizations based on configuration
            if self.output_config.get('show_person_detection', True):
                # Draw persons first
                result_frame = self.person_module.draw_persons(
                    result_frame, 
                    tracked_persons,
                    self.output_config.get('show_person_id', True)
                )
            
            # Calculate and show FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time
            
            # Get video progress information if available
            progress_info = self.camera_module.get_camera_progress()
            
            # Add FPS, camera info, and GPU status to the frame
            info_line = f"FPS: {self.fps:.1f} | Camera: {self.camera_module.get_active_camera_name()}"
            
            # Add GPU status
            if self.using_gpu:
                if self.using_gpu_face and self.using_gpu_person:
                    info_line += " | GPU: Face+Person"
                elif self.using_gpu_face:
                    info_line += " | GPU: Face only"
                elif self.using_gpu_person:
                    info_line += " | GPU: Person only"
            else:
                info_line += " | GPU: Disabled"
                
            # Add stub status if using stub
            if USING_STUB:
                info_line += " | USING FALLBACK FACE DETECTION"
            
            # Add progress information for video files
            if progress_info:
                info_line += f" | Progress: {progress_info['progress_percent']:.1f}% ({progress_info['current_frame']}/{progress_info['total_frames']})"
            
            cv2.putText(
                result_frame,
                info_line,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Add number of tracked persons and faces
            info_line2 = f"Tracked Persons: {len(tracked_persons)} | Tracked Faces: {len(tracked_faces)}"
            cv2.putText(
                result_frame,
                info_line2,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Initialize video writer if it hasn't been created yet
            if self.video_writer is None and frame is not None:
                h, w = result_frame.shape[:2]
                self.output_video_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path,
                    fourcc,
                    self.output_video_fps,
                    self.output_video_size
                )
                print(f"Initialized video output: {self.output_video_path} ({w}x{h} @ {self.output_video_fps}fps)")
            
            # Write frame to output video
            if self.video_writer is not None:
                self.video_writer.write(result_frame)
            
            # Show the result
            if self.output_config.get('show_video', True):
                cv2.imshow('Face and Person Tracker', result_frame)
                
            # Save detected faces if configured
            if self.output_config.get('save_detections', False) and tracked_faces:
                self._save_detected_faces(frame, tracked_faces)
                
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self._switch_to_next_camera()
        
        # Cleanup
        self.stop()
    
    def _save_detected_faces(self, frame, tracked_faces):
        """
        Save detected faces to the output directory.
        
        Args:
            frame (numpy.ndarray): Input frame
            tracked_faces (dict): Dictionary of tracked faces
        """
        output_dir = self.output_config.get('output_dir', './detected_faces')
        
        # First identify the initial face ID for each person
        person_to_initial_face = {}
        for face_id, face_data in tracked_faces.items():
            person_id = face_data.get('person_id', None)
            if person_id is not None:
                if person_id in person_to_initial_face:
                    if face_id < person_to_initial_face[person_id]:
                        person_to_initial_face[person_id] = face_id
                else:
                    person_to_initial_face[person_id] = face_id
                    
        for face_id, face_data in tracked_faces.items():
            # Only save faces that are currently visible
            if not face_data.get('visible', True) and face_data.get('frames_since_seen', 0) > 3:
                continue
                
            bbox = face_data['bbox'].astype(int)
            
            # Get person ID if available
            person_id = face_data.get('person_id', None)
            
            # For consistent filenames, use the initial face ID for this person if available
            display_id = face_id
            if person_id is not None and person_id in person_to_initial_face:
                display_id = person_to_initial_face[person_id]
            else:
                # If no person ID, use face ID directly
                person_id = face_id
            
            # Extract face region
            if (bbox[1] < bbox[3] and bbox[0] < bbox[2] and
                bbox[1] >= 0 and bbox[0] >= 0 and 
                bbox[3] < frame.shape[0] and bbox[2] < frame.shape[1]):
                
                face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if face_img.size == 0:
                    continue
                    
                # Save face image with person ID and consistent face ID in filename
                timestamp = int(time.time())
                filename = f"{output_dir}/person_{person_id}_face_{display_id}_{timestamp}.jpg"
                cv2.imwrite(filename, face_img)
    
    def _switch_to_next_camera(self):
        """Switch to the next available camera"""
        camera_list = self.camera_module.get_camera_list()
        if len(camera_list) <= 1:
            return
            
        current_camera = self.camera_module.get_active_camera_name()
        current_index = camera_list.index(current_camera) if current_camera in camera_list else -1
        next_index = (current_index + 1) % len(camera_list)
        next_camera = camera_list[next_index]
        
        print(f"Switching from camera '{current_camera}' to '{next_camera}'")
        self.camera_module.set_active_camera(next_camera)
    
    def stop(self):
        """Stop the face tracker application"""
        print("Stopping Face Tracker Application...")
        self.running = False
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video output saved to {self.output_video_path}")
        
        # Stop all cameras
        self.camera_module.stop_all_cameras()
        
        # Close database connection if open
        if self.database is not None:
            self.database.close()
        
        # Close all windows
        cv2.destroyAllWindows()
        
        print(f"Processed {self.frame_count} frames at {self.fps:.1f} FPS")
    
    def _try_reidentify_by_appearance(self, track_id, person_data, person_to_face_map):
        """
        Try to reidentify a person by appearance features when face is not visible.
        Enhanced to better handle people with same clothes after occlusion.
        
        Args:
            track_id (int): Tracking ID of the person
            person_data (dict): Person tracking data
            person_to_face_map (dict): Mapping of person IDs to face IDs
        """
        # Skip if person already has face/person ID or doesn't have appearance features
        if person_data.get('face_id') is not None or 'appearance' not in person_data or person_data['appearance'] is None:
            return
            
        best_similarity = 0.40  # Adjusted threshold for better matching of same clothes
        best_history_id = None
        
        # First, try more recent history entries (prioritize recent occlusions)
        recent_history_entries = [(history_id, history) for history_id, history in self.person_to_face_history.items()
                                 if self.frame_count - history['last_seen'] < 300]  # About 10 seconds at 30fps
        
        # Then try all history entries if no recent match is found
        all_history_entries = list(self.person_to_face_history.items())
        
        # Try recent entries first
        for history_id, history in recent_history_entries:
            # Skip if no appearance data or no person ID
            if 'appearance' not in history or history['appearance'] is None or history['person_id'] is None:
                continue
                
            # Skip if this person still has an active track
            if history_id in self.person_module.tracked_persons:
                continue
                
            # Compare appearances
            try:
                similarity = self.person_module._compare_appearances(person_data['appearance'], history['appearance'])
                
                # Check if the history entry was from an occluded person
                was_occluded = history.get('occlusion_status', '') in ['fully_occluded', 'partially_occluded']
                
                # Boost scores for occluded tracks with good clothing matches 
                if was_occluded and similarity > 0.5:
                    similarity += min(0.15, (similarity - 0.5) * 0.3)
                
                # If similarity is good, consider it the same person
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_history_id = history_id
            except Exception as e:
                print(f"Error comparing appearances: {e}")
        
        # If no match found in recent history, try all history
        if best_history_id is None and recent_history_entries != all_history_entries:
            for history_id, history in all_history_entries:
                # Skip if already checked in recent history
                if history_id in [h[0] for h in recent_history_entries]:
                    continue
                    
                # Skip if no appearance data or no person ID
                if 'appearance' not in history or history['appearance'] is None or history['person_id'] is None:
                    continue
                    
                # Skip if this person still has an active track
                if history_id in self.person_module.tracked_persons:
                    continue
                    
                # Compare appearances with a slightly higher threshold for older entries
                try:
                    similarity = self.person_module._compare_appearances(person_data['appearance'], history['appearance'])
                    
                    # If similarity is good, consider it the same person
                    # Use a higher threshold for older history entries to prevent false matches
                    if similarity > best_similarity + 0.05:
                        best_similarity = similarity
                        best_history_id = history_id
                except Exception as e:
                    print(f"Error comparing appearances: {e}")
        
        # If we found a good match, use the historical IDs
        if best_history_id is not None:
            history = self.person_to_face_history[best_history_id]
            
            # Log if this was likely an occlusion recovery
            was_occluded = history.get('occlusion_status', '') in ['fully_occluded', 'partially_occluded']
            if was_occluded:
                print(f"Recovered occluded person: Track {track_id} matched with occluded history {best_history_id} (score: {best_similarity:.2f})")
            
            # Copy face ID and person ID from history
            person_data['face_id'] = history['face_id']
            self.person_module.tracker_to_face_map[track_id] = history['face_id']
            
            if history['person_id'] is not None:
                person_data['person_id'] = history['person_id']
                self.person_module.tracker_to_person_map[track_id] = history['person_id']
                
                # Update database with this association if we have a database
                if self.database is not None and not USING_STUB:
                    self.database.map_face_to_person(
                        history['face_id'], 
                        history['person_id']
                    )
            
            # Update the history record with this track - preserve occlusion status if present
            occlusion_status = history.get('occlusion_status', None)
            self.person_to_face_history[track_id] = {
                'face_id': history['face_id'],
                'person_id': history['person_id'],
                'last_seen': self.frame_count,
                'appearance': person_data['appearance']
            }
            
            # Copy occlusion status if present
            if occlusion_status is not None:
                self.person_to_face_history[track_id]['occlusion_status'] = occlusion_status


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Face Tracker Application')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    args = parser.parse_args()
    
    app = FaceTrackerApp(args.config)
    app.output_video_path = args.output
    app.start()


if __name__ == "__main__":
    main() 