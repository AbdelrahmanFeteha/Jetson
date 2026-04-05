import os

# Ground station backend base URL
API_BASE = os.getenv("GS_API_BASE", "http://localhost:5000")

# Polling interval (seconds)
POLL_INTERVAL = float(os.getenv("GS_POLL_INTERVAL", "1.0"))

# If true, post telemetry updates (optional feature)
ENABLE_TELEMETRY = os.getenv("GS_ENABLE_TELEMETRY", "1") == "1"

# Sample assets for local simulation
SAMPLE_IMAGE_PATH = os.getenv("GS_SAMPLE_IMAGE", "assets/sample.jpg")