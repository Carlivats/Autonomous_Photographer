import sys
import time
import numpy as np
import cv2
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500

model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
model = IMX500(model_path)
intrinsics = model.network_intrinsics
picam2 = Picamera2(model.camera_num)

# Frame Constants
WIDTH, HEIGHT = 640, 480
FRAME_CX, FRAME_CY = WIDTH // 2, HEIGHT // 2

def draw_composition_guides(request):
    metadata = request.get_metadata()
    np_outputs = model.get_outputs(metadata, add_batch=True)
    
    with MappedArray(request, "main") as m:
        # Draw Frame Center Guide
        cv2.drawMarker(m.array, (FRAME_CX, FRAME_CY), (255, 255, 255), 
                       cv2.MARKER_CROSS, 20, 1)

        if np_outputs is not None:
            boxes, scores, labels = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            
            for i, score in enumerate(scores):
                if score > 0.5:  # Confidence Threshold
                    # Convert coordinates to pixel space
                    x, y, w, h = model.convert_inference_coords(boxes[i], metadata, picam2)
                    
                    # Calculate Subject Center
                    subj_cx = x + (w // 2)
                    subj_cy = y + (h // 2)
                    
                    # Calculate Offset (Error)
                    error_x = subj_cx - FRAME_CX
                    error_y = subj_cy - FRAME_CY
                    
                    is_balanced = abs(error_x) < 30 and abs(error_y) < 30
                    color = (0, 255, 0) if is_balanced else (0, 0, 255)
                    
                    # Bounding Box
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)
                    
                    # Line from Frame Center to Subject Center
                    cv2.line(m.array, (FRAME_CX, FRAME_CY), (subj_cx, subj_cy), color, 1)
                    
                    # Labels
                    label_name = intrinsics.labels[int(labels[i])]
                    text = f"{label_name} | Offset: X:{error_x} Y:{error_y}"
                    cv2.putText(m.array, text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Picamera2 Startup
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (WIDTH, HEIGHT)},
    controls={"FrameRate": intrinsics.inference_rate}
)

picam2.pre_callback = draw_composition_guides
model.show_network_fw_progress_bar()
picam2.start(config, show_preview=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()