import time

from config import API_BASE, POLL_INTERVAL, ENABLE_TELEMETRY, SAMPLE_IMAGE_PATH
from api import get_command
from inspector import run_inspection_cycle


def main():
    print("=== Jetson Client Simulator ===")
    print(f"API_BASE: {API_BASE}")
    print("Polling for begin_inspection...\n")

    last_handled = None

    while True:
        try:
            data = get_command(API_BASE)
            cmd = data.get("command", {})
            cmd_type = cmd.get("type", "none")

            if cmd_type == "begin_inspection" and last_handled != "begin_inspection":
                print("[CMD] begin_inspection received")
                last_handled = "begin_inspection"

                run_inspection_cycle(
                    api_base=API_BASE,
                    sample_image_path=SAMPLE_IMAGE_PATH,
                    enable_telemetry=ENABLE_TELEMETRY,
                )

            if cmd_type == "none":
                last_handled = None

        except Exception as e:
            print(f"[WARN] {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()