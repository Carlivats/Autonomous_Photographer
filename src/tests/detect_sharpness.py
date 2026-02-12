import sys
import time
import numpy as np
import cv2
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500
from src.get_sharpness_score import get_subject_sharpness

model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
model = IMX500(model_path)
intrinsics = model.network_intrinsics
picam2 = Picamera2(model.camera_num)

def draw_boxes(request):
    metadata = request.get_metadata()
    np_outputs = model.get_outputs(metadata, add_batch=True)
    
    if np_outputs is not None:
        with MappedArray(request, "main") as m:
            boxes, scores, labels = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            
            for i, score in enumerate(scores):
                if score > 0.6:
                    # 1. Get Coordinates
                    x, y, w, h = model.convert_inference_coords(boxes[i], metadata, picam2)
                    label_name = intrinsics.labels[int(labels[i])]
                    
                    # 2. Calculate Sharpness ONLY for this detection
                    sharpness = get_subject_sharpness(m.array, x, y, w, h)
                    
                    # 3. Visual Feedback Logic
                    # If sharpness is low (< 100), use Red. If high, use Green.
                    color = (0, 255, 0) if sharpness > 200 else (0, 0, 255)
                    
                    # Draw Box
                    cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)
                    
                    # 4. Add Label + Sharpness Score
                    text = f"{label_name}: {score:.2f} | Sharp: {int(sharpness)}"
                    cv2.putText(m.array, text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Standard Picamera2 Startup
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)},
    controls={"FrameRate": intrinsics.inference_rate}
)

picam2.pre_callback = draw_boxes
model.show_network_fw_progress_bar()
picam2.start(config, show_preview=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()