# 🚗 Speed Breaker Detection & Driver Alert System

A vision-based ADAS (Advanced Driver Assistance System) that detects
speed breakers ahead of a vehicle using a dash-mounted camera,
estimates their distance, and provides an early warning to the driver.

## 🎯 Project Objective

The system is designed to identify upcoming speed breakers before the
vehicle reaches them and alert the driver in advance.

The system provides:

- 🚧 Speed breaker detection using computer vision
- 📏 Distance estimation from the detected speed breaker
- 🔊 Audio warning that becomes faster as the speed breaker gets closer
- ⚠️ Visual warning displayed to the driver
- 📹 Real-time processing using a camera feed

## 🧠 How It Works

```text
Camera Input
     ↓
Video Frame Processing
     ↓
Speed Breaker Detection
     ↓
Bounding Box Detection
     ↓
Distance Estimation
     ↓
Driver Alert
     ↓
Audio + Visual Warning