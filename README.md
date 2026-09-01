# 🚗 Speed Breaker Detection & Driver Alert System

A vision-based **ADAS (Advanced Driver Assistance System)** that detects
speed breakers ahead of a vehicle using a dash-mounted camera, estimates
their distance, and provides an early warning to the driver.

## 🎯 Project Objective

The system is designed to identify upcoming speed breakers before the
vehicle reaches them and alert the driver in advance.

### Key Features

- 🚧 Speed breaker detection using **YOLOv8**
- 📏 Monocular distance estimation from the detected speed breaker
- ⏱️ Time-to-arrival estimation using distance and vehicle speed
- 🔊 Adaptive audio warning — beeping becomes faster as the speed breaker gets closer
- ⚠️ Visual warning / speedometer-style HUD
- 📹 Real-time processing from camera or video input
- 🧠 Temporal smoothing using consecutive-frame confirmation to reduce false alarms

## 🧠 How It Works

```text
Camera / Video Input
        ↓
Video Frame Processing
        ↓
YOLOv8 Speed Breaker Detection
        ↓
Bounding Box Extraction
        ↓
Distance Estimation
        ↓
Time-to-Arrival Calculation
        ↓
Warning Logic
        ↓
Audio + Visual Driver Alert
```

## 🤖 Machine Learning Model

The project uses **YOLOv8** for object detection.

The detector is trained to identify:

- `Speed-Bump`
- `Rumble Strip`

The training pipeline uses image augmentation to improve robustness to
real-road conditions such as changes in lighting, camera angle, scale,
and image appearance.

## 📊 Model Performance

Validation results from the trained model:

| Metric | Score |
|---|---:|
| **mAP50** | **0.917** |
| **mAP50-95** | **0.423** |
| **Precision** | **0.895** |
| **Recall** | **0.911** |

The model achieved a **0.917 mAP50** with **0.911 recall** on the validation
set.

## 📏 Distance Estimation

The system estimates distance using a calibrated monocular camera.

The primary distance cue is the detected speed breaker's bounding-box width,
combined with an assumed lane width and camera focal length.

```text
Distance ≈ (Focal Length × Assumed Lane Width) / Bounding Box Width
```

A one-time camera calibration is used to determine the effective focal
length.

## ⏱️ Driver Alert Logic

The estimated distance is combined with the current vehicle speed to
calculate the approximate **time to arrival (TTA)**.

The warning becomes more urgent as TTA decreases:

- No warning when the speed breaker is sufficiently far away
- Visual warning when it enters the warning window
- Faster audio beeps as the speed breaker gets closer
- Urgent warning inside the critical TTA window

The current implementation can use simulated vehicle speed and is structured
so that real vehicle speed can later be integrated through **OBD-II**.

## 🛠️ Technologies Used

- **Python**
- **YOLOv8 / Ultralytics**
- **OpenCV**
- **NumPy**
- **PyGame**
- **PyYAML**
- **Computer Vision**
- **Object Detection**
- **Monocular Distance Estimation**

## 📁 Project Structure

```text
speedbreaker-detection/
│
├── train_speed_breaker_yolo.py
├── speed_breaker_alert_system.py
├── distance_estimator.py
├── README.md
├── requirements.txt
└── yolov8n.pt
```

Training outputs and generated prediction files are kept outside the main
source-code structure where appropriate.

## ▶️ Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the YOLOv8 model

The training script contains the dataset configuration, augmentation,
training parameters, validation, and model-saving logic.

```bash
python train_speed_breaker_yolo.py --data "$DATA"
```

### 3. Run detection on a video

Use the trained `best.pt` weights and provide a video source:

```bash
yolo predict model="path/to/best.pt" source="path/to/video.mp4" conf=0.25 device=mps save=True
```

### 4. Run the driver alert system

```bash
python speed_breaker_alert_system.py   --source "path/to/video.mp4"   --weights "path/to/best.pt"   --sim-speed 40
```

Press **Q** to exit the real-time display.

## 🎥 Demo

A test video was processed through the trained YOLOv8 detector, producing
annotated detection output.

Example output:

```text
speed_bump_detection_result.mp4
```

## 🔮 Future Improvements

- Integrate real vehicle speed through **OBD-II**
- Improve distance accuracy with better camera calibration
- Add stereo/depth-camera support
- Expand the dataset with more road, lighting, and weather conditions
- Deploy on an embedded device for real-time in-vehicle use
- Add additional road-hazard classes

## ⚠️ Disclaimer

This is an academic / prototype ADAS project. Distance estimates and alerts
should not be treated as a replacement for a certified vehicle safety system.

## 👨‍💻 Project Highlights

This project demonstrates practical experience with:

- End-to-end object detection model training
- Dataset-based computer vision
- YOLOv8 model validation and evaluation
- Image augmentation for real-world robustness
- Real-time video inference
- Monocular distance estimation
- Time-to-arrival based alert logic
- Audio and visual driver-warning systems
