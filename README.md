# Enhanced Multi-Modal Person Tracking System

This system provides robust person tracking and identification through the fusion of multiple biometric modalities:

1. **Face Recognition**: Deep learning-based face detection and recognition
2. **Gait Recognition**: Analysis of walking patterns and motion
3. **Pose-based Recognition**: Skeleton tracking for body movement analysis 
4. **Appearance Features**: Clothing and visual attributes
5. **Occlusion Handling**: Prediction and management of hidden or partially visible people

## Features

### Multi-Modal Fusion
- Weighted fusion combining face, gait, and pose-based identifications
- Adaptive weighting based on recognition quality
- Temporal consistency analysis to prevent ID switches
- Confidence scoring system for reliable identification

### Gait Recognition
- Extract distinctive walking patterns as biometric signatures
- Analysis of stride length, step frequency, and body movement
- Persistent database for matching people across sessions
- Works even when face is not visible or at a distance

### Pose Estimation
- Full skeleton tracking with joint angle analysis
- Enhanced identification through distinctive movement patterns
- Improved handling of occlusions with pose-based prediction
- Additional accuracy when combined with gait analysis

### Advanced Occlusion Handling
- Detection of partial and full occlusions
- Trajectory prediction for temporarily hidden people
- Maintenance of identity during and after occlusions
- Visual indicators for occlusion status

### Extensible Architecture
- Modular design allowing easy addition of new recognition methods
- Configuration-based customization
- Separate databases for each modality
- Comprehensive visualization options

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy
- PyYAML
- For face recognition: InsightFace (or stub version)
- For pose estimation: OpenPose or MoveNet models
- For person detection: YOLOv5 and ByteTrack/BoTSORT

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure the system by editing `config.yml`

3. Run the application:
   ```
   python face_tracker_app.py --config config.yml --output results.mp4
   ```

## Configuration Options

The system is highly configurable through the `config.yml` file:

```yaml
cameras:
  sources:
    - name: Default Camera
      source: 0
      type: webcam
  default_camera: Default Camera

face_recognition:
  model: buffalo_l
  detection_threshold: 0.5
  recognition_threshold: 0.5
  use_gpu: true

gait_recognition:
  sequence_length: 20
  min_track_length: 10
  similarity_threshold: 0.7
  db_path: gait_database.pkl

pose_estimation:
  model_path: models/pose
  confidence_threshold: 0.5
  use_gpu: true

fusion:
  face_weight: 0.6
  gait_weight: 0.25
  pose_weight: 0.15
  appearance_weight: 0.2
  fusion_threshold: 0.55
  adaptive_weights: true

occlusion:
  overlap_threshold: 0.5
  min_area_ratio: 0.3

output:
  show_video: true
  show_person_detection: true
  show_pose: true
  show_gait: true
  show_fusion: true
  show_occlusions: true
```

## Architecture

The system uses a modular architecture with these key components:

1. **Camera Module**: Handles video input from multiple sources
2. **Face Module**: Face detection and recognition
3. **Person Tracking Module**: Person detection and tracking
4. **Gait Module**: Gait analysis and signature matching
5. **Pose Module**: Skeleton tracking and analysis
6. **Fusion Module**: Multi-modal biometric fusion
7. **Occlusion Handler**: Occlusion detection and handling

Each module is independent and can be enabled/disabled as needed.

## License

[MIT License](LICENSE) 