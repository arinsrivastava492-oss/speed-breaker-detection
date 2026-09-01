
import argparse
import time
from collections import deque

import cv2
import numpy as np
import pygame

from distance_estimator import CameraCalibration, estimate_distance_m, time_to_arrival_s

CONFIDENCE_THRESHOLD = 0.45        # min detector confidence to accept a box
CONSECUTIVE_FRAMES_REQUIRED = 3    # temporal smoothing -> fewer false alarms
TTA_WARNING_S = 6.0                # start visual warning at this many seconds out
TTA_URGENT_S = 2.5                 # fastest beep / red state inside this window
BEEP_MIN_INTERVAL_S = 0.15         # fastest allowed beep repetition
BEEP_MAX_INTERVAL_S = 1.2          # slowest beep repetition (far away)


class BeepAlert:
    """Generates a beep whose repetition rate increases as the bump gets closer."""

    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        self._beep_sound = self._generate_beep_wave()
        self._last_beep_time = 0.0

    @staticmethod
    def _generate_beep_wave(freq_hz=1500, duration_s=0.12, volume=0.6):
        sample_rate = 44100
        n_samples = int(sample_rate * duration_s)
        t = np.linspace(0, duration_s, n_samples, False)
        wave = np.sin(freq_hz * t * 2 * np.pi)
        # short fade in/out to avoid audio clicks
        fade = np.linspace(0, 1, int(n_samples * 0.1))
        wave[: len(fade)] *= fade
        wave[-len(fade):] *= fade[::-1]
        audio = (wave * volume * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(audio)

    def update(self, tta_s: float):
        """Call every frame. Plays a beep at a rate proportional to urgency."""
        if tta_s is None or tta_s > TTA_WARNING_S:
            return  # too far away / no detection -> silent

        # Map time-to-arrival to a beep interval: closer = faster beeping
        urgency = np.clip(
            (TTA_WARNING_S - tta_s) / (TTA_WARNING_S - TTA_URGENT_S), 0.0, 1.0
        )
        interval = BEEP_MAX_INTERVAL_S - urgency * (BEEP_MAX_INTERVAL_S - BEEP_MIN_INTERVAL_S)

        now = time.time()
        if now - self._last_beep_time >= interval:
            self._beep_sound.play()
            self._last_beep_time = now


def draw_speedometer_hud(frame, speed_kmh: float, tta_s, distance_m):
    """
    Draws a speedometer-style HUD in the corner of the frame with a
    warning ring/icon that activates when a speed breaker is approaching.
    This mimics what would appear as an overlay on/near a real digital
    instrument cluster or an aftermarket dash display.
    """
    h, w = frame.shape[:2]
    center = (w - 130, h - 130)
    radius = 100

    alert_active = tta_s is not None and tta_s <= TTA_WARNING_S
    urgent = tta_s is not None and tta_s <= TTA_URGENT_S

    # Base gauge ring
    ring_color = (80, 80, 80)
    cv2.circle(frame, center, radius, ring_color, 8)

    # Speed needle arc (0-180 km/h mapped to 270 degrees of the ring)
    max_speed = 180
    angle = 135 + (speed_kmh / max_speed) * 270
    needle_color = (0, 200, 255)
    cv2.ellipse(frame, center, (radius, radius), 0, 135, min(angle, 405), needle_color, 8)

    # Speed text
    cv2.putText(frame, f"{int(speed_kmh)}", (center[0] - 35, center[1] + 15),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(frame, "km/h", (center[0] - 30, center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if alert_active:
        # Pulsing warning ring: red when urgent, amber when approaching
        warn_color = (0, 0, 255) if urgent else (0, 165, 255)
        pulse = int(8 + 6 * abs(np.sin(time.time() * (8 if urgent else 3))))
        cv2.circle(frame, center, radius + 12, warn_color, pulse)

        # Warning banner + distance readout
        label = "SPEED BREAKER AHEAD" if not urgent else "SLOW DOWN NOW"
        cv2.rectangle(frame, (20, 20), (480, 90), warn_color, -1)
        cv2.putText(frame, label, (35, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        if distance_m is not None:
            cv2.putText(frame, f"{distance_m:.0f} m  |  {tta_s:.1f}s",
                        (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame


def get_vehicle_speed_kmh(sim_speed):
    """
    Placeholder for live vehicle speed.
    Replace this with a real OBD-II read (python-obd) for deployment.
    """
    return sim_speed


def run(source, weights_path, sim_speed, calib_focal_px):
    from ultralytics import YOLO  # imported here so --help doesn't require it installed

    model = YOLO(weights_path)
    calibration = CameraCalibration(
        focal_length_px=calib_focal_px, frame_width_px=1280, frame_height_px=720
    )
    beeper = BeepAlert()

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    detection_history = deque(maxlen=CONSECUTIVE_FRAMES_REQUIRED)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        speed_kmh = get_vehicle_speed_kmh(sim_speed)

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        best_box = None
        best_conf = 0.0
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

        detection_history.append(best_box is not None)
        confirmed = sum(detection_history) >= CONSECUTIVE_FRAMES_REQUIRED

        tta_s = None
        distance_m = None
        if confirmed and best_box is not None:
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            bbox_width_px = x2 - x1
            distance_m = estimate_distance_m(bbox_width_px, calibration)
            tta_s = time_to_arrival_s(distance_m, speed_kmh)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"speed_breaker {best_conf:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        beeper.update(tta_s)
        frame = draw_speedometer_hud(frame, speed_kmh, tta_s, distance_m)

        cv2.imshow("Speed Breaker Alert System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time speed breaker detection + alert")
    parser.add_argument("--source", default="0", help="Video file path or camera index (0)")
    parser.add_argument("--weights", required=True, help="Path to trained YOLOv8 .pt weights")
    parser.add_argument("--sim-speed", type=float, default=40.0,
                         help="Simulated vehicle speed in km/h (replace with OBD-II for real use)")
    parser.add_argument("--calib-focal-px", type=float, default=1000.0,
                         help="Camera focal length in pixels from distance_estimator.calibrate()")
    args = parser.parse_args()

    run(args.source, args.weights, args.sim_speed, args.calib_focal_px)
