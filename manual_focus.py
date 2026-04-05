# simple_manual_focus_preview.py
import time
from Focuser import Focuser
from JetsonCamera import Camera

I2C_BUS = 9       # try 9 first, if not then try 10, 7, or 8
FOCUS_VALUE = 0

def main():
    print(f"Opening camera...")
    camera = Camera()
    camera.start_preview()

    print(f"Creating focuser on I2C bus {I2C_BUS}...")
    focuser = Focuser(I2C_BUS)

    time.sleep(1.0)

    print(f"Setting focus to {FOCUS_VALUE}...")
    focuser.set(Focuser.OPT_FOCUS, FOCUS_VALUE)

    print("Preview running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nClosing...")
    finally:
        camera.stop_preview()
        camera.close()

if __name__ == "__main__":
    main()
