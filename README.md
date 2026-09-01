# Speed Breaker Detection & Driver Alert System

A vision-based ADAS (Advanced Driver Assistance System) module that detects
speed breakers (speed bumps) ahead of the vehicle using a dash-mounted camera,
estimates how far away they are, and warns the driver with:

- An audio beep that gets faster as the bump gets closer
- A visual warning indicator overlaid on (or next to) the speedometer

---

## 1. Why camera-based, not accelerometer-based

You asked for a system that "signals the driver about the **incoming** speed
breaker so the driver can lower speed." That means the warning has to arrive
**before** the wheels hit the bump.

| Approach | Detects bump... | Useful for advance warning? |
|---|---|---|
| Accelerometer/IMU | The instant you hit it | No — too late, it's reactive not predictive |
| GPS + crowdsourced map | Only bumps already logged by someone else | Partial — misses new/unmapped bumps |
| **Camera + object detection** | **50–150m ahead, while approaching** | **Yes — this is the only proactive method** |

So this system is built around a fine-tuned **YOLOv8** object detector running
on frames from a forward-facing dash camera. (A hybrid version that fuses
camera detections with GPS-tagged bump locations for redundancy is described
in section 5, and is the best long-term design.)

---

## 2. System pipeline

```
Dash camera (frame @ 15-30 FPS)
        │
        ▼
YOLOv8 detector  ──► bounding box for "speed_breaker" class
        │
        ▼
Distance estimator (bbox height + camera calibration → meters)
        │
        ▼
Alert controller (distance + current vehicle speed → time-to-arrival)
        │
        ├──► Audio: beep, interval shortens as TTA (time-to-arrival) shrinks
        └──► Display: warning icon + colored arc/ring on speedometer HUD
```

Files in this project:

| File | Purpose |
|---|---|
| `train_speed_breaker_yolo.py` | Fine-tunes YOLOv8 on a labeled speed-breaker dataset |
| `distance_estimator.py` | Converts a detected bounding box into an estimated distance in meters |
| `speed_breaker_alert_system.py` | Real-time inference loop: camera/video in → beep + HUD out |
| `requirements.txt` | Python dependencies |

---

## 3. On "as accurate as possible" — what actually drives accuracy

I can hand you a fully working pipeline right now, but I want to be upfront
about the one thing that determines real-world accuracy more than any code
change: **the training dataset**.

A model is only as good as what it's trained on. To get a genuinely reliable
detector you need:

1. **Volume**: 3,000–8,000+ labeled images minimum for a single-class
   detector to generalize well; 15,000+ for production-grade robustness.
2. **Diversity**: painted vs. unpainted bumps, faded paint, rumble strips,
   potholes vs. bumps (a common false-positive source), different road
   surfaces (asphalt/concrete/gravel), lighting (day/dusk/night/headlights),
   weather (rain, fog, wet-road glare), camera mounting height/angle,
   different countries' bump styles (India's speed breakers look very
   different from US speed humps or UK "sleeping policemen").
3. **Good negatives**: images of normal road *without* bumps, and especially
   *near-miss* confusers (manhole covers, shadows, tar patches, zebra
   crossings, potholes) so the model learns what a bump is **not**.
4. **Annotation quality**: tight, consistent bounding boxes. Sloppy labels
   cap your accuracy no matter how good the model architecture is.

### Where to get data
- Roboflow Universe has several public "speed bump" / "speed breaker"
  detection datasets you can download in YOLO format and merge.
- Kaggle has a few Indian-roads speed-breaker datasets (useful for
  your context given your paint style/road type).
- Best results: collect your own footage from the actual routes/vehicle
  you'll deploy on, and label ~1,000–2,000 of those yourself with a tool
  like Roboflow or CVAT — even a small amount of *your own* road/camera
  data mixed into a larger public dataset meaningfully boosts real-world
  accuracy, because it adapts the model to your camera's lens, mounting
  angle, and local road markings.

The training script below is written so you just point it at a dataset
folder in YOLO format and it handles the rest (transfer learning from
COCO-pretrained weights, augmentation, validation, mAP reporting).

---

## 4. Real car integration path

This repo gives you the detection + alert *logic*. Wiring it into an actual
car has three realistic integration tiers:

1. **Aftermarket dash unit (easiest, no car modification)**: Raspberry Pi 4 /
   Jetson Nano + USB or Pi camera, mounted on the dashboard, with a small
   display and speaker. Runs `speed_breaker_alert_system.py` directly. This
   is the fastest path to a working prototype.
2. **OBD-II integration (for real vehicle speed)**: instead of assuming
   speed, read live vehicle speed via an ELM327 OBD-II adapter (`python-obd`
   library) over Bluetooth/USB so the time-to-arrival calculation uses your
   *actual* speed, not an estimate.
3. **OEM infotainment/instrument cluster integration**: pushing the alert
   icon onto the actual speedometer display (not just an external screen)
   requires access to the vehicle's CAN bus and the manufacturer's HMI
   software — this is only accessible to OEMs/Tier-1 suppliers or via
   specialized (and often unofficial/warranty-voiding) CAN-bus injection
   hardware. I've built the HUD as a **standalone display module** that
   mimics a speedometer with a warning ring, which is the realistic
   approach for an aftermarket/prototype system.

---

## 5. Suggested accuracy improvements beyond v1

- **Sensor fusion**: add a cheap IMU. Use it not for *detecting* bumps ahead,
  but to *confirm* past detections (did we actually feel a bump where the
  camera predicted one?) and auto-flag false positives/negatives for
  retraining — a feedback loop that improves the model over time.
- **Temporal smoothing**: require a detection to persist across 3-5
  consecutive frames before alerting, cutting single-frame false positives
  (shadows, glare) substantially.
- **GPS caching**: once a bump is confirmed by the camera, log its GPS
  coordinate. On future passes, use the cached location as a prior to boost
  detection confidence at that spot even in poor visibility.
- **Model size vs. speed tradeoff**: YOLOv8n (nano) for Raspberry Pi/low-power
  boards, YOLOv8s/m if you have a Jetson or better GPU — bigger models are
  more accurate but need more compute to hit real-time FPS.
