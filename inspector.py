import base64
import os
from datetime import datetime
import time
import cv2

from api import post_inspection, post_telemetry
from Focuser import Focuser
from JetsonCamera import Camera


I2C_BUS = 9
FOCUS_VALUE = 400
CAPTURE_PATH = "captures/inspection.jpg"


def _b64_from_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def capture_image_with_manual_focus(output_path: str, i2c_bus: int = I2C_BUS, focus_value: int = FOCUS_VALUE):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    camera = None
    try:
        # Open camera
        camera = Camera()

        # Use the working focus method
        focuser = Focuser(i2c_bus)
        time.sleep(1.0)
        focuser.set(Focuser.OPT_FOCUS, focus_value)
        print(f"[INFO] Focus set to {focus_value}")

        # Small wait so lens settles
        time.sleep(1.0)

        # Grab one frame
        frame = camera.getFrame()

        if frame is None:
            raise RuntimeError("Failed to capture frame from camera")

        ok = cv2.imwrite(output_path, frame)
        if not ok:
            raise RuntimeError(f"Failed to save image to {output_path}")

        print(f"[OK] Image saved: {output_path}")
        return output_path

    finally:
        if camera is not None:
            camera.close()


def run_inspection_cycle(api_base: str, sample_image_path: str = CAPTURE_PATH, enable_telemetry: bool = True):
    inspection_id = f"insp_{int(time.time())}"
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Capture fresh image using working focus code
    capture_image_with_manual_focus(sample_image_path)

    # Optional telemetry
    if enable_telemetry:
        try:
            post_telemetry(api_base, {
                "inspection_id": inspection_id,
                "inspection_stage": "completed",
                "progress_percent": 100,
                "timestamp": timestamp,
            })
        except Exception:
            pass

    if not os.path.exists(sample_image_path):
        raise FileNotFoundError(f"Image not found: {sample_image_path}")

    image_b64 = _b64_from_file(sample_image_path)

    # Fake static results for now
    payload = {
        "inspection_id": inspection_id,
        "timestamp": timestamp,
        "height_cm": 120,
        "has_crack": False,
        "confidence": 0.15,
        "image": image_b64,
    }

    response = post_inspection(api_base, payload)

    print(f"[OK] Inspection uploaded instantly: {inspection_id}")
    return response
