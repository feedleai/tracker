# Face and Person Tracking Application

A comprehensive modular application for tracking persons and their faces using:
- **YOLOv8** for person detection
- **BoTSORT** from Ultralytics for person tracking
- **InsightFace** for face detection and recognition
- **SQLite** for storing face embeddings and maintaining persistent identities

The application maintains consistent IDs for people across frames by associating person tracks with face identities, and even across sessions using the SQLite database. It is designed to be configurable and can easily handle multiple camera sources.

## Features

- Person detection and tracking using YOLOv8 and BoTSORT
- Face detection and recognition with InsightFace
- Persistent person identification across sessions using SQLite
- Association of persons with their face identities
- Support for multiple cameras (webcam, IP cameras, RTSP streams)
- Easy configuration via YAML file
- Real-time tracking with unique IDs for each person and face
- Camera switching capability
- Optional saving of detected faces with person IDs

## Requirements

- Python 3.6+
- OpenCV
- InsightFace
- Ultralytics (YOLOv8)
- NumPy
- PyYAML
- ONNX Runtime
- SQLite3 (included with Python)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/face-person-tracker.git
   cd face-person-tracker
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application with default configuration:
   ```
   python face_tracker_app.py
   ```

2. Run with a custom configuration file:
   ```
   python face_tracker_app.py --config custom_config.yml
   ```

3. During execution:
   - Press 'q' to quit the application
   - Press 's' to switch between available cameras

## Configuration

The application can be configured using a YAML file (`config.yml` by default). Here's an example configuration:

```yaml
face_recognition:
  detection_size: (640, 640)
  recognition_threshold: 0.5
  top_k: 1

person_tracking:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.7
  max_age: 30
  min_hits: 3

database:
  db_path: "face_database.db"
  use_db: true
  min_detections_to_store: 3
  update_period: 5  # Update database every N frames

cameras:
  # Local camera
  - name: "default_camera"
    source: 0
    enabled: true
  
  # Example of additional camera sources
  # - name: "front_door"
  #   source: "rtsp://username:password@192.168.1.100:554/stream"
  #   enabled: false
  # 
  # - name: "back_yard"
  #   source: "http://192.168.1.101:8080/video"
  #   enabled: false

output:
  show_video: true
  save_detections: false
  output_dir: "./detected_faces"
  show_person_detection: true
  show_face_recognition: true
  show_person_id: true
```

### Configuration Options

#### Face Recognition

- `detection_size`: The size of the detection input (width, height)
- `recognition_threshold`: Similarity threshold for face recognition (0.0-1.0)
- `top_k`: Number of top matches to consider

#### Person Tracking

- `model_path`: Path to YOLOv8 model (can be model name like 'yolov8n.pt' or a local path)
- `confidence_threshold`: Confidence threshold for person detection (0.0-1.0)
- `iou_threshold`: IoU threshold for NMS in person detection
- `max_age`: Maximum frames a track can be lost before removal
- `min_hits`: Minimum detections before a track is confirmed

#### Database

- `db_path`: Path to the SQLite database file
- `use_db`: Whether to use the database for persistent person identification
- `min_detections_to_store`: Minimum detections before storing a face in the database
- `update_period`: How often to update the database (every N frames)

#### Cameras

- `name`: Unique identifier for the camera
- `source`: Camera source (0 for default webcam, URL for IP cameras)
- `enabled`: Whether the camera is enabled

#### Output

- `show_video`: Whether to display the video window
- `save_detections`: Whether to save detected faces
- `output_dir`: Directory to save detected faces
- `show_person_detection`: Whether to show person detection boxes
- `show_face_recognition`: Whether to show face detection boxes
- `show_person_id`: Whether to show person IDs instead of track IDs

## How It Works

1. **Person Detection**: YOLOv8 detects people in each frame
2. **Person Tracking**: BoTSORT tracks detected people across frames
3. **Face Detection**: InsightFace detects faces in the frame
4. **Face Recognition**: Detected faces are compared with previously seen faces
5. **Database Matching**: Faces are matched against stored embeddings in the SQLite database
6. **Person-Face Association**: The system associates each tracked person with their recognized face
7. **Consistent IDs**: People maintain the same ID even across sessions thanks to the database
8. **ID Persistence**: Even if a face is temporarily not visible, the person maintains the same ID

This approach allows for robust tracking of people even when their faces are not always visible or when they reappear in later sessions.

### Understanding the Display

- **Green bounding box (Person)**: Person with recognized face and database ID
- **Yellow bounding box (Person)**: Person with detected face but not yet matched in database
- **Blue bounding box (Person)**: Person without detected face
- **Green bounding box (Face)**: Face matched to a known person in the database
- **Red bounding box (Face)**: New face not yet matched in the database

## Adding New Cameras

To add a new camera, edit the `config.yml` file and add a new entry under the `cameras` section:

```yaml
cameras:
  - name: "new_camera"
    source: "rtsp://username:password@192.168.1.100:554/stream"
    enabled: true
```

The application supports various camera sources:
- Local webcams (use an integer index, e.g., 0 for the default webcam)
- IP cameras (HTTP URLs)
- RTSP streams
- Video files

## Project Structure

- `face_tracker_app.py`: Main application entry point
- `face_module.py`: Face detection and recognition module
- `person_tracking_module.py`: Person detection and tracking module
- `camera_module.py`: Camera handling module
- `config_module.py`: Configuration handling module
- `database_module.py`: Database handling for persistent face storage
- `config.yml`: Configuration file
- `requirements.txt`: Python dependencies
- `face_database.db`: SQLite database for storing face embeddings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the face recognition model
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for person detection and tracking
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [SQLite](https://www.sqlite.org/) for database functionality 