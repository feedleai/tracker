import os
import yaml


class ConfigModule:
    def __init__(self, config_path="config.yml"):
        """
        Initialize the configuration module.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """
        Load configuration from the YAML file.
        
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            print(f"Configuration file not found at {self.config_path}. Using default configuration.")
            return self._create_default_config()
            
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate the config
            if not self._validate_config(config):
                print("Invalid configuration file. Using default configuration.")
                return self._create_default_config()
                
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return self._create_default_config()
    
    def save_config(self, config=None):
        """
        Save the configuration to the YAML file.
        
        Args:
            config (dict, optional): Configuration to save. If None, saves the current config.
            
        Returns:
            bool: Success or failure
        """
        if config is not None:
            self.config = config
            
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get_face_recognition_config(self):
        """Get face recognition configuration"""
        return self.config.get('face_recognition', {})
    
    def get_person_tracking_config(self):
        """Get person tracking configuration"""
        return self.config.get('person_tracking', {})
    
    def get_database_config(self):
        """Get database configuration"""
        return self.config.get('database', {})
    
    def get_cameras_config(self):
        """Get cameras configuration"""
        return self.config.get('cameras', [])
    
    def get_output_config(self):
        """Get output configuration"""
        return self.config.get('output', {})
    
    def _validate_config(self, config):
        """
        Validate the configuration structure.
        
        Args:
            config (dict): Configuration to validate
            
        Returns:
            bool: Whether the configuration is valid
        """
        # Check if required sections exist
        required_sections = ['face_recognition', 'cameras', 'output']
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
                
        # Validate camera configs
        cameras = config.get('cameras', [])
        if not isinstance(cameras, list):
            print("Cameras configuration must be a list")
            return False
            
        for i, camera in enumerate(cameras):
            if not isinstance(camera, dict):
                print(f"Camera #{i+1} configuration must be a dictionary")
                return False
                
            if 'name' not in camera or 'source' not in camera:
                print(f"Camera #{i+1} must have 'name' and 'source' fields")
                return False
        
        return True
    
    def _create_default_config(self):
        """
        Create a default configuration.
        
        Returns:
            dict: Default configuration
        """
        default_config = {
            'face_recognition': {
                'detection_size': '(800, 800)',
                'recognition_threshold': 0.6,
                'top_k': 1,
                'use_gpu': True
            },
            'person_tracking': {
                'model_path': 'yolov8n.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.7,
                'max_age': 30,
                'min_hits': 3,
                'use_gpu': True
            },
            'database': {
                'db_path': 'face_database.db',
                'use_db': True,
                'min_detections_to_store': 3,
                'update_period': 5
            },
            'cameras': [
                {
                    'name': 'default_camera',
                    'source': 0,
                    'enabled': True
                }
            ],
            'output': {
                'show_video': True,
                'save_detections': False,
                'output_dir': './detected_faces',
                'show_person_detection': True,
                'show_face_recognition': True,
                'show_person_id': True
            }
        }
        
        # Save the default configuration
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(default_config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error creating default configuration file: {e}")
            
        return default_config 