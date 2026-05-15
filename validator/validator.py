import cv2
import numpy as np
import polars


def is_black_frame(frame, threshold=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    avg_brightness = np.mean(gray)

    return avg_brightness < threshold


def validate_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)

    success, frame = cap.read()

    cap.release()

    if not success:
        return {
            "valid": False,
            "reason": "Could not read frame"
        }

    if is_black_frame(frame):
        return {
            "valid": False,
            "reason": "Black frame detected"
        }

    return {
        "valid": True,
        "reason": "Frame looks valid"
    }