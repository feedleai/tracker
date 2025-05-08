import os
import sqlite3
import numpy as np
import pickle
import time


class DatabaseModule:
    def __init__(self, db_path="face_database.db"):
        """
        Initialize the database module for storing face embeddings.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the database connection and create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    num_detections INTEGER DEFAULT 1
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    embedding BLOB,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            # Add face_id_mapping table to store the relationship between face IDs and person IDs
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_id_mapping (
                    face_id INTEGER PRIMARY KEY,
                    person_id INTEGER,
                    last_seen TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            self.conn.commit()
            print(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            if self.conn:
                self.conn.close()
            # Create in-memory database as fallback
            self.conn = sqlite3.connect(':memory:')
            self.cursor = self.conn.cursor()
            self._initialize_database()
            
    def add_or_update_person(self, face_embedding, person_id=None):
        """
        Add a new person or update an existing person with a new face embedding.
        
        Args:
            face_embedding (numpy.ndarray): The face embedding vector
            person_id (int, optional): Person ID if known, otherwise will create new or match
            
        Returns:
            int: The person ID
        """
        try:
            current_time = time.time()
            
            # If no person_id provided, try to match with existing embeddings
            if person_id is None:
                person_id = self.find_matching_person(face_embedding)
            
            # If still no match, create a new person
            if person_id is None:
                self.cursor.execute(
                    "INSERT INTO persons (first_seen, last_seen) VALUES (?, ?)",
                    (current_time, current_time)
                )
                person_id = self.cursor.lastrowid
                
                # Store the embedding without checking uniqueness for new persons
                embedding_blob = pickle.dumps(face_embedding)
                self.cursor.execute(
                    "INSERT INTO face_embeddings (person_id, embedding, timestamp) VALUES (?, ?, ?)",
                    (person_id, embedding_blob, current_time)
                )
            else:
                # Update existing person's last_seen time
                self.cursor.execute(
                    "UPDATE persons SET last_seen = ?, num_detections = num_detections + 1 WHERE id = ?",
                    (current_time, person_id)
                )
                
                # Check if this embedding is unique enough to be worth storing
                self.cursor.execute(
                    "SELECT embedding FROM face_embeddings WHERE person_id = ? ORDER BY timestamp DESC LIMIT 5",
                    (person_id,)
                )
                
                embeddings_rows = self.cursor.fetchall()
                is_unique = True
                
                # Skip empty result check - a person should always have at least one embedding
                for row in embeddings_rows:
                    stored_embedding = pickle.loads(row[0])
                    similarity = self._calculate_similarity(face_embedding, stored_embedding)
                    
                    # If similarity is above 0.9, this embedding is too similar to existing ones
                    if similarity > 0.9:
                        is_unique = False
                        break
                
                # Only store the embedding if it provides new information
                if is_unique:
                    embedding_blob = pickle.dumps(face_embedding)
                    self.cursor.execute(
                        "INSERT INTO face_embeddings (person_id, embedding, timestamp) VALUES (?, ?, ?)",
                        (person_id, embedding_blob, current_time)
                    )
            
            self.conn.commit()
            return person_id
            
        except sqlite3.Error as e:
            print(f"Error adding/updating person: {e}")
            self.conn.rollback()
            return None
    
    def find_matching_person(self, face_embedding, threshold=0.6):
        """
        Find a matching person for the given face embedding.
        Enhanced for long-term reidentification with optimized thresholds.
        
        Args:
            face_embedding (numpy.ndarray): The face embedding to match
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            int: Matched person ID or None if no match found
        """
        try:
            # Get all person IDs
            self.cursor.execute("""
                SELECT p.id, p.num_detections, p.last_seen 
                FROM persons p
                ORDER BY p.num_detections DESC
            """)
            person_data = [(row[0], row[1], row[2]) for row in self.cursor.fetchall()]
            
            best_similarity = 0
            best_person_id = None
            current_time = time.time()
            
            # Pre-process input embedding
            normalized_embedding = self._normalize_embedding(face_embedding)
            
            # For each person, get their embeddings and compare
            # Process persons with more detections first to prefer established identities
            for person_id, num_detections, last_seen in person_data:
                try:
                    # Calculate how long since this person was last seen (in hours)
                    hours_since_seen = (current_time - last_seen) / 3600.0 if last_seen else 0
                    
                    # For long-term reidentification, get more embeddings than usual
                    # This increases the chance of finding a match with different appearances
                    embedding_limit = 15 if hours_since_seen > 1.0 else 10
                    
                    # For very long absences, we increase our coverage even more
                    if hours_since_seen > 12.0:
                        embedding_limit = 25  # Get even more embeddings for people gone for 12+ hours
                    
                    # Get more recent embeddings for this person
                    self.cursor.execute(
                        "SELECT embedding FROM face_embeddings WHERE person_id = ? ORDER BY timestamp DESC LIMIT ?",
                        (person_id, embedding_limit)
                    )
                    
                    embeddings_rows = self.cursor.fetchall()
                    if not embeddings_rows:
                        continue
                    
                    # Calculate similarity with each embedding
                    similarities = []
                    for row in embeddings_rows:
                        try:
                            stored_embedding = pickle.loads(row[0])
                            normalized_stored = self._normalize_embedding(stored_embedding)
                            similarity = self._calculate_similarity(normalized_embedding, normalized_stored)
                            similarities.append(similarity)
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")
                            continue
                    
                    # Sort similarities and get average of top N (if available)
                    # For long-term reidentification, we take a smaller "best subset" to handle appearance changes
                    if similarities:
                        similarities.sort(reverse=True)
                        
                        # For long absences, we're more flexible - look at just the best matches
                        # instead of averaging many matches which might include changed appearances
                        if hours_since_seen > 1.0:
                            top_n = min(2, len(similarities))  # Take only best 2 matches for long absences
                        else:
                            top_n = min(3, len(similarities))  # Use 3 for short absences
                        
                        avg_similarity = sum(similarities[:top_n]) / top_n
                        
                        # For long-term ID, detection bonus is more important to maintain identity consistency
                        # This helps establish consistent recognition on reappearance
                        detection_bonus = min(0.05, 0.001 * num_detections)
                        
                        # For people who've been seen many times, give an extra bonus to maintain stability
                        if num_detections > 100:
                            detection_bonus += 0.02
                        
                        # For long absences, we're more lenient with matching threshold
                        # to account for appearance changes (clothing, hair, etc.)
                        threshold_adjustment = min(0.07, hours_since_seen * 0.005)
                        effective_threshold = max(0.4, threshold - threshold_adjustment)
                        
                        adjusted_similarity = avg_similarity + detection_bonus
                        
                        if adjusted_similarity > best_similarity:
                            best_similarity = adjusted_similarity
                            best_person_id = person_id
                except Exception as e:
                    print(f"Error processing person {person_id}: {e}")
                    continue
            
            # Return the best match if it's above the effective threshold
            # For long-term identification, we're more lenient with very frequent faces
            if best_similarity >= threshold:
                return best_person_id
            
            return None
            
        except sqlite3.Error as e:
            print(f"Error finding matching person: {e}")
            return None
    
    def _normalize_embedding(self, embedding):
        """
        Normalize an embedding vector to unit length.
        
        Args:
            embedding (numpy.ndarray): Face embedding vector
            
        Returns:
            numpy.ndarray: Normalized embedding vector
        """
        try:
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
            return embedding
        except Exception as e:
            print(f"Error normalizing embedding: {e}")
            return embedding
    
    def get_person_info(self, person_id):
        """
        Get information about a person.
        
        Args:
            person_id (int): The person ID
            
        Returns:
            dict: Person information or None if not found
        """
        try:
            self.cursor.execute(
                "SELECT id, first_seen, last_seen, num_detections FROM persons WHERE id = ?",
                (person_id,)
            )
            
            row = self.cursor.fetchone()
            if not row:
                return None
                
            return {
                'id': row[0],
                'first_seen': row[1],
                'last_seen': row[2],
                'num_detections': row[3]
            }
            
        except sqlite3.Error as e:
            print(f"Error getting person info: {e}")
            return None
    
    def get_all_persons(self):
        """
        Get all persons in the database.
        
        Returns:
            list: List of person dictionaries
        """
        try:
            self.cursor.execute(
                "SELECT id, first_seen, last_seen, num_detections FROM persons ORDER BY id"
            )
            
            persons = []
            for row in self.cursor.fetchall():
                persons.append({
                    'id': row[0],
                    'first_seen': row[1],
                    'last_seen': row[2],
                    'num_detections': row[3]
                })
                
            return persons
            
        except sqlite3.Error as e:
            print(f"Error getting all persons: {e}")
            return []
            
    def get_recent_embeddings(self, person_id, limit=5):
        """
        Get the most recent embeddings for a person.
        
        Args:
            person_id (int): The person ID
            limit (int): Maximum number of embeddings to retrieve
            
        Returns:
            list: List of embeddings (numpy arrays)
        """
        try:
            self.cursor.execute(
                "SELECT embedding FROM face_embeddings WHERE person_id = ? ORDER BY timestamp DESC LIMIT ?",
                (person_id, limit)
            )
            
            embeddings = []
            for row in self.cursor.fetchall():
                embedding = pickle.loads(row[0])
                embeddings.append(embedding)
                
            return embeddings
            
        except sqlite3.Error as e:
            print(f"Error getting recent embeddings: {e}")
            return []
    
    def _calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two face embeddings.
        Assumes embeddings are already normalized to unit length.
        
        Args:
            embedding1 (numpy.ndarray): First face embedding
            embedding2 (numpy.ndarray): Second face embedding
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Both embeddings should already be flattened and normalized
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(embedding1, embedding2)
            
            # Ensure the result is within valid range [-1, 1]
            similarity = max(-1.0, min(1.0, similarity))
            
            # Convert to positive similarity score [0, 1]
            return (similarity + 1) / 2 if similarity < 0 else similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def map_face_to_person(self, face_id, person_id):
        """
        Map a face ID to a person ID in the database.
        
        Args:
            face_id (int): The face ID from face detection
            person_id (int): The person ID from the database
            
        Returns:
            bool: Success or failure
        """
        try:
            current_time = time.time()
            
            # Check if this face ID already has a mapping
            self.cursor.execute(
                "SELECT person_id FROM face_id_mapping WHERE face_id = ?",
                (face_id,)
            )
            
            result = self.cursor.fetchone()
            if result:
                # Update existing mapping
                self.cursor.execute(
                    "UPDATE face_id_mapping SET person_id = ?, last_seen = ? WHERE face_id = ?",
                    (person_id, current_time, face_id)
                )
            else:
                # Create new mapping
                self.cursor.execute(
                    "INSERT INTO face_id_mapping (face_id, person_id, last_seen) VALUES (?, ?, ?)",
                    (face_id, person_id, current_time)
                )
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"Error mapping face to person: {e}")
            self.conn.rollback()
            return False
            
    def get_person_id_from_face_id(self, face_id):
        """
        Get the person ID associated with a face ID.
        
        Args:
            face_id (int): The face ID
            
        Returns:
            int: Person ID or None if not found
        """
        try:
            self.cursor.execute(
                "SELECT person_id FROM face_id_mapping WHERE face_id = ?",
                (face_id,)
            )
            
            result = self.cursor.fetchone()
            if result:
                return result[0]
            return None
            
        except sqlite3.Error as e:
            print(f"Error getting person ID from face ID: {e}")
            return None
            
    def get_all_face_person_mappings(self):
        """
        Get all face ID to person ID mappings.
        
        Returns:
            dict: Dictionary with face IDs as keys and person IDs as values
        """
        try:
            self.cursor.execute(
                "SELECT face_id, person_id FROM face_id_mapping"
            )
            
            mappings = {}
            for row in self.cursor.fetchall():
                mappings[row[0]] = row[1]
                
            return mappings
            
        except sqlite3.Error as e:
            print(f"Error getting face-person mappings: {e}")
            return {}
            
    def update_person_for_face_embedding(self, face_embedding, face_id, force_update=False):
        """
        Update person information for a face embedding and face ID.
        Enhanced for better long-term recognition.
        
        Args:
            face_embedding (numpy.ndarray): The face embedding
            face_id (int): Face ID to map
            force_update (bool): Force update the person even if it's already mapped
            
        Returns:
            int: Matched or created person ID
        """
        try:
            # Check if this face ID already has a person ID
            if not force_update:
                self.cursor.execute(
                    "SELECT person_id FROM face_id_mapping WHERE face_id = ?",
                    (face_id,)
                )
                result = self.cursor.fetchone()
                
                if result is not None and result[0] is not None:
                    existing_person_id = result[0]
                    
                    # Update last seen time
                    current_time = time.time()
                    self.cursor.execute(
                        "UPDATE face_id_mapping SET last_seen = ? WHERE face_id = ?",
                        (current_time, face_id)
                    )
                    
                    # Update last_seen in persons table too
                    self.cursor.execute(
                        "UPDATE persons SET last_seen = ? WHERE id = ?",
                        (current_time, existing_person_id)
                    )
                    
                    # Check if this embedding provides new information
                    # Get recent embeddings for this person
                    self.cursor.execute(
                        "SELECT embedding FROM face_embeddings WHERE person_id = ? ORDER BY timestamp DESC LIMIT 10",
                        (existing_person_id,)
                    )
                    
                    embeddings_rows = self.cursor.fetchall()
                    is_unique = True
                    
                    # Only save this embedding if it's sufficiently different from existing ones
                    for row in embeddings_rows:
                        stored_embedding = pickle.loads(row[0])
                        similarity = self._calculate_similarity(face_embedding, stored_embedding)
                        
                        # If similarity is above 0.9, this embedding is too similar to existing ones
                        if similarity > 0.9:
                            is_unique = False
                            break
                    
                    # Store this new embedding only if it provides new information
                    if is_unique:
                        embedding_blob = pickle.dumps(face_embedding)
                        self.cursor.execute(
                            "INSERT INTO face_embeddings (person_id, embedding, timestamp) VALUES (?, ?, ?)",
                            (existing_person_id, embedding_blob, current_time)
                        )
                    
                    # Always update num_detections to track frequency
                    self.cursor.execute(
                        "UPDATE persons SET num_detections = num_detections + 1 WHERE id = ?",
                        (existing_person_id,)
                    )
                    
                    self.conn.commit()
                    return existing_person_id
            
            # Find the best matching person for this face embedding
            # Use a lower threshold for long-term tracking to better handle reappearance
            person_id = self.find_matching_person(face_embedding, threshold=0.5)
            
            # If no match, create a new person with the same ID as the face ID
            if person_id is None:
                person_id = self.add_or_update_person(face_embedding, face_id)
            else:
                # Check if this embedding is sufficiently different from existing ones
                self.cursor.execute(
                    "SELECT embedding FROM face_embeddings WHERE person_id = ? ORDER BY timestamp DESC LIMIT 10",
                    (person_id,)
                )
                
                embeddings_rows = self.cursor.fetchall()
                is_unique = True
                best_similarity = 0
                
                for row in embeddings_rows:
                    stored_embedding = pickle.loads(row[0])
                    similarity = self._calculate_similarity(face_embedding, stored_embedding)
                    best_similarity = max(best_similarity, similarity)
                    
                    # If similarity is above 0.9, this embedding is too similar to existing ones
                    if similarity > 0.9:
                        is_unique = False
                        break
                
                # Store new embeddings in these cases:
                # 1. If it's unique (not too similar to existing embeddings)
                # 2. If it's similar but not identical (0.7-0.9 range) and we don't have many embeddings
                # This helps build a more robust embedding set for long-term identification
                should_store = is_unique or (
                    best_similarity > 0.7 and 
                    best_similarity < 0.9 and 
                    len(embeddings_rows) < 20
                )
                
                if should_store:
                    self.add_or_update_person(face_embedding, person_id)
                else:
                    # Even if we don't store the embedding, update last_seen time and detection count
                    current_time = time.time()
                    self.cursor.execute(
                        "UPDATE persons SET last_seen = ?, num_detections = num_detections + 1 WHERE id = ?",
                        (current_time, person_id)
                    )
                    
            # Map the face ID to the person ID
            self.map_face_to_person(face_id, person_id)
            
            return person_id
            
        except sqlite3.Error as e:
            print(f"Error updating person for face embedding: {e}")
            self.conn.rollback()
            return None 