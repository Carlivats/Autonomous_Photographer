import sys
import time
import numpy as np
import cv2
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500
from src.get_sharpness_score import get_subject_sharpness

# Thresholds
SHARP_THRESHOLD = 100.0
STABILITY_REQUIRED = 5

# Model Setup
model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
model = IMX500(model_path)
intrinsics = model.network_intrinsics
picam2 = Picamera2(model.camera_num)

subject_trackers = {}

def flag_blur(request):
    global subject_trackers
    metadata = request.get_metadata()
    np_outputs = model.get_outputs(metadata, add_batch=True)
    
    current_frame_detections = set()

    if np_outputs is not None:
        with MappedArray(request, "main") as m:
            boxes, scores, labels = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            
            for i, score in enumerate(scores):
                if score > 0.6:
                    label_idx = int(labels[i])
                    label_name = intrinsics.labels[label_idx]
                    current_frame_detections.add(label_idx)
                    
                    # 1. Get Coordinates & Calculate Sharpness ROI
                    x, y, w, h = model.convert_inference_coords(boxes[i], metadata, picam2)
                    sharpness = get_subject_sharpness(m.array, x, y, w, h)
                    
                    # 2. Logic: Sharpness + Motion Stability
                    if sharpness > SHARP_THRESHOLD:
                        # Increment stability counter for this subject
                        subject_trackers[label_idx] = subject_trackers.get(label_idx, 0) + 1
                    else:
                        # Reset if it gets blurry (Motion or Focus hunt)
                        subject_trackers[label_idx] = 0

                    # 3. Determine Status and Label Color
                    is_ready = subject_trackers[label_idx] >= STABILITY_REQUIRED
                    
                    if is_ready:
                        color = (0, 255, 0)  # Green: Sharp and Still
                        status_text = "READY TO CAPTURE"
                    elif sharpness > SHARP_THRESHOLD:
                        color = (255, 255, 0) # Yellow: Sharp but moving/stabilizing
                        status_text = f"Stabilizing... ({subject_trackers[label_idx]}/{STABILITY_REQUIRED})"
                    else:
                        color = (0, 0, 255)  # Red: Blurry
                        status_text = "Waiting for Focus..."

                    # 4. Visual Feedback
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)
                    
                    # Header Text
                    header = f"{label_name} | Sharpness: {int(sharpness)}"
                    cv2.putText(m.array, header, (x, y - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # Sub-status Text
                    cv2.putText(m.array, status_text, (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Cleanup trackers for objects no longer in view
    subject_trackers = {k: v for k, v in subject_trackers.items() if k in current_frame_detections}

# Standard Picamera2 Startup
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)},
    controls={"FrameRate": intrinsics.inference_rate}
)

picam2.pre_callback = flag_blur
model.show_network_fw_progress_bar()
picam2.start(config, show_preview=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()