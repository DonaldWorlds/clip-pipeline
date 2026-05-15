import subprocess
import json
import cv2



def run_ffprobe(video_path):
    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    return json.loads(result.stdout)


def run_terminal_command(command):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def extract_frame(video_path, frame_number=30):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if success:
        output_path = "frame.jpg"
        cv2.imwrite(output_path, frame)

        return output_path

    return None