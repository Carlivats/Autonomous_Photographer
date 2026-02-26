import sys
import time
import numpy as np
import cv2
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from picamera2 import Picamera2, MappedArray
from picamera2.devices.imx500 import IMX500
from adafruit_servokit import ServoKit

model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
model = IMX500(model_path)
intrinsics = model.network_intrinsics
picam2 = Picamera2(model.camera_num)

# Setup Servos
kit = ServoKit(channels=16)
pan_angle = 90
tilt_angle = 90
kit.servo[0].angle = pan_angle
kit.servo[1].angle = tilt_angle

# Frame Constants
WIDTH, HEIGHT = 640, 480
FRAME_CX, FRAME_CY = WIDTH // 2, HEIGHT // 2

def centered_symetry(request):
    global pan_angle, tilt_angle
    
    metadata = request.get_metadata()
    np_outputs = model.get_outputs(metadata, add_batch=True)
    
    with MappedArray(request, "main") as m:
        # Draw Frame Center Guide
        cv2.drawMarker(m.array, (FRAME_CX, FRAME_CY), (255, 255, 255), 
                       cv2.MARKER_CROSS, 20, 1)

        if np_outputs is not None:
            boxes, scores, labels = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            
            # We track the first subject found with > 0.5 confidence
            found_idx = -1
            for i, score in enumerate(scores):
                if score > 0.5:
                    found_idx = i
                    break
            
            if found_idx != -1:
                x, y, w, h = model.convert_inference_coords(boxes[found_idx], metadata, picam2)
                
                # Calculate Subject Center
                subj_cx = x + (w // 2)
                subj_cy = y + (h // 2)
                
                # Calculate Error (Distance from center)
                error_x = subj_cx - FRAME_CX
                error_y = subj_cy - FRAME_CY
                
                # Servo Update Logic
                # Gain: Controls how "aggressive" the movement is (lower is smoother)
                gain = 0.05 
                deadzone = 15 # Ignore errors smaller than 15 pixels
                
                if abs(error_x) > deadzone:
                    # Subtract error because if subject is to the RIGHT (positive error), 
                    # we need to decrease pan angle to look right.
                    pan_angle -= error_x * gain
                    pan_angle = max(0, min(180, pan_angle)) # Clamp between 0-180
                
                if abs(error_y) > deadzone:
                    # If subject is BELOW center (positive error), 
                    # we need to increase tilt angle to look down.
                    tilt_angle += error_y * gain
                    tilt_angle = max(0, min(180, tilt_angle))

                # Write to Servos
                kit.servo[0].angle = pan_angle
                kit.servo[1].angle = tilt_angle

                # Visuals
                color = (0, 255, 0) if (abs(error_x) < 30 and abs(error_y) < 30) else (0, 0, 255)
                cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)
                cv2.line(m.array, (FRAME_CX, FRAME_CY), (subj_cx, subj_cy), color, 1)
                
                text = f"P:{int(pan_angle)} T:{int(tilt_angle)} | Error:[{int(error_x)}, {int(error_y)}]"
                cv2.putText(m.array, text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Startup
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (WIDTH, HEIGHT)},
    controls={"FrameRate": intrinsics.inference_rate}
)

picam2.pre_callback = centered_symetry
picam2.start(config, show_preview=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()