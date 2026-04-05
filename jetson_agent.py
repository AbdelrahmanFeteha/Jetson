#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
jetson_agent.py  (Headless command-driven inspection)

- Polls Ground Station GET /command
- When command.type == "begin_inspection":
    - capture one frame from camera (no UI, no keypress)
    - run classification; if crack -> run segmentation + severity
    - POST /inspection with REQUIRED fields:
        inspection_id, timestamp, height_cm, has_crack, confidence
    - optional: includes base64-encoded JPEG under "image" (backend supports it)
- Focus is manual (you set it physically/software elsewhere to 200). This script does NOT touch I2C.

Requires your existing:
- api.py  (get_command, post_inspection)
- segmentation.py  (MobileNetV3UNet)
- best_segmentation_model.pth
- best_classification_model.pth

Run example:
  export GS_API_BASE="http://192.168.1.10:5000"
  python3 jetson_agent.py --sensor-id 0 --width 960 --height 540 --fr 30
"""

import os
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime, timezone
from Focuser import Focuser

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from skimage.morphology import skeletonize

from api import get_command, post_inspection
from segmentation import MobileNetV3UNet


# -----------------------------
# Camera (GStreamer)
# -----------------------------
def gstreamer_pipeline(sensor_id=0, width=960, height=540, framerate=30, flip=0):
    return (
        f"nvarguscamerasrc sensor_id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )


def open_camera(sensor_id: int, width: int, height: int, fr: int, flip: int) -> cv2.VideoCapture:
    pipe = gstreamer_pipeline(sensor_id=sensor_id, width=width, height=height, framerate=fr, flip=flip)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    return cap


def grab_stable_frame(cap: cv2.VideoCapture, warm_frames: int = 10) -> np.ndarray:
    """
    Read a few frames to avoid the first-frame exposure/awb weirdness.
    """
    frame = None
    for _ in range(max(1, warm_frames)):
        ret, f = cap.read()
        if ret:
            frame = f
    if frame is None:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


# -----------------------------
# Geometry / severity
# -----------------------------
def extract_skeleton_features(mask_u8_255: np.ndarray):
    """
    mask_u8_255: HxW uint8 (0 or 255)
    returns: length_px, avg_width_px, max_width_px, n_branches
    """
    binary = (mask_u8_255 > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return 0, 0.0, 0.0, 0

    skel = skeletonize(binary > 0).astype(np.uint8)
    length_px = int(np.count_nonzero(skel))
    if length_px == 0:
        return 0, 0.0, 0.0, 0

    dist = cv2.distanceTransform(binary * 255, cv2.DIST_L2, 5)
    ys, xs = np.where(skel > 0)
    widths = dist[ys, xs] * 2.0 if len(xs) else np.array([0.0], dtype=np.float32)
    avg_w = float(np.mean(widths)) if widths.size else 0.0
    max_w = float(np.max(widths)) if widths.size else 0.0

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neigh = cv2.filter2D(skel, -1, kernel)
    branch = (neigh >= 3) & (skel > 0)
    n_branches = int(np.count_nonzero(branch))

    return length_px, avg_w, max_w, n_branches


def classify_severity(length_px, avg_width_px, max_width_px, n_branches, mask_area_px, W, H):
    """
    Simple normalized severity score (tunable).
    Returns (score, level).
    """
    alpha, beta, gamma, delta = 0.3, 0.3, 0.3, 0.1

    max_area = W * H * 0.5
    max_len = W * 1.5
    max_w = 50.0
    max_b = 20.0

    A_n = mask_area_px / max(max_area, 1)
    L_n = length_px / max(max_len, 1)
    W_n = max_width_px / max(max_w, 1e-6)
    B_n = n_branches / max(max_b, 1)

    score = alpha * A_n + beta * L_n + gamma * W_n + delta * B_n

    if score < 0.30:
        level = "Minor"
    elif score < 0.60:
        level = "Moderate"
    else:
        level = "Severe"
    return float(score), level


# -----------------------------
# Models
# -----------------------------
def load_segmentation_model(path: str, device: str):
    model = MobileNetV3UNet(out_channels=1, pretrained=False).to(device)
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            raise
    model.eval()
    return model


def build_classification_model(device: str):
    from torchvision.models import MobileNet_V3_Small_Weights
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    return m.to(device).eval()


def load_classification_weights(model, path: str, device: str):
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            raise
    model.eval()
    return model


CLS_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

SEG_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def infer_frame(
    frame_bgr: np.ndarray,
    seg_model,
    cls_model,
    device: str,
    seg_threshold: float,
    class_threshold: float,
    min_mask_area_px: int,
):
    """
    Returns:
      prob_crack, has_crack, severity_score, severity_level, features(dict), mask_area_px(int)
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # Classification
    x = CLS_TF(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cls_model(x).squeeze(1)
        prob_noncrack = torch.sigmoid(logits).cpu().item()

    # Your training convention: prob_noncrack -> crack prob = 1 - prob_noncrack
    prob_crack = 1.0 - prob_noncrack

    # If not confident enough, stop here
    if prob_crack < class_threshold:
        return prob_crack, False, None, None, None, 0, None

    # Segmentation
    s = SEG_TF(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = seg_model(s)
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()

    mask_small = (prob > seg_threshold).astype(np.uint8) * 255
    H, W = frame_bgr.shape[:2]
    mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 127).astype(np.uint8) * 255

    mask_area = int(np.count_nonzero(mask))
    if mask_area < min_mask_area_px:
        # classification said crack, but segmentation too small/noisy
        return prob_crack, False, None, None, None, mask_area, mask

    length_px, avg_w, max_w, n_branches = extract_skeleton_features(mask)
    sev_score, sev_level = classify_severity(length_px, avg_w, max_w, n_branches, mask_area, W, H)

    features = {
        "mask_area_px": mask_area,
        "length_px": int(length_px),
        "avg_width_px": float(avg_w),
        "max_width_px": float(max_w),
        "branch_points": int(n_branches),
    }

    return prob_crack, True, sev_score, sev_level, features, mask_area, mask


def bgr_to_base64_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default=os.getenv("GS_API_BASE", "").strip(),
                    help="Ground Station base URL, e.g. http://192.168.1.10:5000 (or set GS_API_BASE env var)")
    ap.add_argument("--poll", type=float, default=0.25, help="Polling interval seconds")

    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fr", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)

    ap.add_argument("--device", default=None, help="cpu or cuda (auto if omitted)")
    ap.add_argument("--seg-model", default="best_segmentation_model.pth")
    ap.add_argument("--class-model", default="best_classification_model.pth")

    ap.add_argument("--seg-threshold", type=float, default=0.30)
    ap.add_argument("--class-threshold", type=float, default=0.50)
    ap.add_argument("--min-mask-area", type=int, default=150, help="min crack pixels after resize")

    ap.add_argument("--height-cm", type=float, default=0.0, help="placeholder until range sensor; required by backend")
    ap.add_argument("--send-image", action="store_true", help="Include base64 image in /inspection payload")
    ap.add_argument("--jpeg-quality", type=int, default=85)

    ap.add_argument("--warm-frames", type=int, default=10)
    
    ap.add_argument("--focus", type=int, default=400, help="Manual focus value (0-1000)")
    ap.add_argument("--i2c-bus", type=int, default=9, help="I2C bus for Arducam focuser")

    args = ap.parse_args()

    if not args.api_base:
        raise SystemExit("ERROR: --api-base is required (or set GS_API_BASE). Example: http://192.168.1.10:5000")

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] API_BASE={args.api_base}")
    print(f"[INFO] device={device}")
    print(f"[INFO] camera: sensor_id={args.sensor_id} {args.width}x{args.height}@{args.fr}")

    # Resolve model paths
    script_dir = Path(__file__).resolve().parent
    seg_path = Path(args.seg_model)
    cls_path = Path(args.class_model)
    if not seg_path.exists():
        seg_path = script_dir / seg_path
    if not cls_path.exists():
        cls_path = script_dir / cls_path
    if not seg_path.exists() or not cls_path.exists():
        raise SystemExit(f"ERROR: model file(s) not found:\n  {seg_path}\n  {cls_path}")

    # Open camera first (more stable on Jetson)
    cap = open_camera(args.sensor_id, args.width, args.height, args.fr, args.flip)
    if not cap.isOpened():
        raise SystemExit("ERROR: Could not open camera via OpenCV GStreamer. (Argus / OpenCV build / pipeline issue)")

    # Grab one frame to ensure camera is actually streaming
    _ = grab_stable_frame(cap, warm_frames=args.warm_frames)
    print("[INFO] Camera streaming OK.")
    
    # Set manual focus once using Arducam focuser
    try:
    	focuser = Focuser(args.i2c_bus)
    	time.sleep(1.0)
    	focuser.set(Focuser.OPT_FOCUS, args.focus)
    	print(f"[INFO] Manual focus set to {args.focus} on I2C bus {args.i2c_bus}")
    	time.sleep(1.0)  # let lens settle
    except Exception as e:
    	print(f"[WARN] Failed to set manual focus: {e}")

    # Load models once
    print("[INFO] Loading models...")
    seg_model = load_segmentation_model(str(seg_path), device)
    cls_model = load_classification_weights(build_classification_model(device), str(cls_path), device)
    print("[INFO] Models loaded.")

    last_handled_cmd_id = None  # simple de-dupe if GS repeats same command dict

    while True:
        try:
            cmd = get_command(args.api_base)
        except Exception as e:
            print(f"[WARN] Ground station unreachable: {e}")
            time.sleep(1.0)
            continue

        command = cmd.get("command", {}) or {}
        cmd_type = command.get("type", "none")

        if cmd_type != "begin_inspection":
            time.sleep(args.poll)
            continue

        # (Optional) crude de-dupe: if GS keeps returning the same command dict, avoid double-run

        # --- Run inspection cycle ---
        try:
            frame = grab_stable_frame(cap, warm_frames=args.warm_frames)
            t0 = time.time()

            prob_crack, has_crack, sev_score, sev_level, features, mask_area, mask = infer_frame(
                frame_bgr=frame,
                seg_model=seg_model,
                cls_model=cls_model,
                device=device,
                seg_threshold=args.seg_threshold,
                class_threshold=args.class_threshold,
                min_mask_area_px=args.min_mask_area,
            )

            dt = time.time() - t0

            # Required fields for backend:
            inspection_id = f"insp_{int(time.time()*1000)}"
            timestamp = datetime.now(timezone.utc).isoformat()
            if mask is not None:
                overlay = frame.copy()
                overlay[mask > 0] = [0, 0, 255]  # RED cracks
            else:
                overlay = frame
            payload = {
                "inspection_id": inspection_id,
                "timestamp": timestamp,
                "height_cm": float(args.height_cm),         # placeholder for now
                "has_crack": bool(has_crack),
                "confidence": float(prob_crack),            # use prob_crack as confidence
                # Extra fields (backend will store them fine):
                "inference_seconds": float(dt),
                "severity_score": (float(sev_score) if sev_score is not None else None),
                "severity_level": sev_level,
                "features": features,
                "mask_area_px": int(mask_area),
                "camera": {
                    "sensor_id": args.sensor_id,
                    "width": args.width,
                    "height": args.height,
                    "fr": args.fr,
                },
                "focus_manual": int(args.focus),  # current fixed focus (manual)
            }

            if args.send_image:
                payload["image"] = bgr_to_base64_jpeg(overlay, quality=args.jpeg_quality)
                cv2.imwrite("debug.jpg", frame)

            resp = post_inspection(args.api_base, payload)
            print(f"[OK] INSPECTION sent id={inspection_id} crack={has_crack} conf={prob_crack:.3f} time={dt:.2f}s resp={resp}")

        except Exception as e:
            print(f"[ERROR] Inspection failed: {e}")

        # brief pause so we don't instantly re-trigger if GS takes a moment to clear state
        time.sleep(0.2)


if __name__ == "__main__":
    main()
