from validator import validate_video_frame
from tools import run_terminal_command


def render_with_retry(command, output_path, max_retries=3):

    for attempt in range(max_retries):

        print(f"Attempt {attempt + 1}")

        run_terminal_command(command)

        validation = validate_video_frame(output_path)

        print(validation)

        if validation["valid"]:
            print("Render successful")
            return True

        print("Validation failed. Retrying...")

    return False