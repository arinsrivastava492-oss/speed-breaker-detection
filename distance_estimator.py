"""
distance_estimator.py
----------------------
Estimates real-world distance (in meters) to a detected speed breaker
from its bounding box size in the camera frame, using the standard
"known object height" pinhole-camera formula:

    distance = (real_world_height * focal_length_px) / bbox_height_px

This needs a one-time camera calibration (see calibrate() below) but
after that requires no extra hardware -- just the existing dash camera.

For higher accuracy, this can later be replaced/augmented with a stereo
camera or a monocular depth-estimation model (e.g. MiDaS), but the
single-camera approach below is sufficient for a real-time embedded
system and is what's used here.
"""

from dataclasses import dataclass


# Typical painted speed breaker height in most countries: ~8-10 cm raised
# section, but what the camera actually resolves well is the painted
# stripe pattern width across the road, which correlates with lane width
# (a much larger, more stable feature at range than the 8-10cm bump height).
# We use lane-width-normalized bbox width as the primary distance cue,
# with bbox height as a secondary/backup cue. Tune ASSUMED_LANE_WIDTH_M
# for your actual road type during calibration.
ASSUMED_LANE_WIDTH_M = 3.0  # standard single-lane width, adjust per region


@dataclass
class CameraCalibration:
    focal_length_px: float   # from calibrate()
    frame_width_px: int
    frame_height_px: int


def calibrate(known_distance_m: float, known_width_m: float, bbox_width_px: float) -> float:
    """
    One-time calibration step. Place a speed breaker (or a marker of known
    width) at a MEASURED distance from the camera, run detection, note the
    bounding box width in pixels, and call this function once to derive
    the camera's effective focal length in pixels.

    Example:
        # Marker is 3.0m wide (one lane), placed 10m from camera,
        # detector reports a 210px wide bounding box at that distance:
        focal_length_px = calibrate(known_distance_m=10.0,
                                     known_width_m=3.0,
                                     bbox_width_px=210)
    """
    return (bbox_width_px * known_distance_m) / known_width_m


def estimate_distance_m(bbox_width_px: float, calibration: CameraCalibration) -> float:
    """
    Given a detected bounding box width (pixels) and the calibration,
    return estimated distance to the speed breaker in meters.
    """
    if bbox_width_px <= 0:
        return float("inf")
    return (calibration.focal_length_px * ASSUMED_LANE_WIDTH_M) / bbox_width_px


def time_to_arrival_s(distance_m: float, vehicle_speed_kmh: float) -> float:
    """
    Converts distance + current vehicle speed into estimated seconds
    until the vehicle reaches the speed breaker. This is what actually
    drives the alert urgency (not raw distance alone), since the same
    distance is far more urgent at 80 km/h than at 20 km/h.
    """
    if vehicle_speed_kmh <= 0.5:
        return float("inf")  # vehicle stationary/near-stationary
    speed_m_s = vehicle_speed_kmh * 1000 / 3600
    return distance_m / speed_m_s
