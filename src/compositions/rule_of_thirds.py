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

# 1. Initialize IMX500 and Camera
model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
model = IMX500(model_path)
intrinsics = model.network_intrinsics
picam2 = Picamera2(model.camera_num)

# 2. Setup Servos
kit = ServoKit(channels=16)
pan_angle, tilt_angle = 90, 90
kit.servo[0].angle, kit.servo[1].angle = pan_angle, tilt_angle

# Frame Constants & Rule of Thirds Points
WIDTH, HEIGHT = 640, 480
THIRD_W = WIDTH // 3
THIRD_H = HEIGHT // 3

# The 4 "Power Points" (Intersections)
POWER_POINTS = [
    (THIRD_W, THIRD_H),         # Top-Left
    (2 * THIRD_W, THIRD_H),     # Top-Right
    (THIRD_W, 2 * THIRD_H),     # Bottom-Left
    (2 * THIRD_W, 2 * THIRD_H)  # Bottom-Right
]

def track_rule_of_thirds(request):
    global pan_angle, tilt_angle
    
    metadata = request.get_metadata()
    np_outputs = model.get_outputs(metadata, add_batch=True)
    
    with MappedArray(request, "main") as m:
        # Draw Rule of Thirds Grid
        cv2.line(m.array, (THIRD_W, 0), (THIRD_W, HEIGHT), (200, 200, 200), 1)
        cv2.line(m.array, (2 * THIRD_W, 0), (2 * THIRD_W, HEIGHT), (200, 200, 200), 1)
        cv2.line(m.array, (0, THIRD_H), (WIDTH, THIRD_H), (200, 200, 200), 1)
        cv2.line(m.array, (0, 2 * THIRD_H), (WIDTH, 2 * THIRD_H), (200, 200, 200), 1)

        if np_outputs is not None:
            boxes, scores = np_outputs[0][0], np_outputs[1][0]
            
            # Find the highest confidence subject
            best_idx = np.argmax(scores) if len(scores) > 0 else -1
            
            if best_idx != -1 and scores[best_idx] > 0.5:
                x, y, w, h = model.convert_inference_coords(boxes[best_idx], metadata, picam2)
                subj_cx, subj_cy = x + (w // 2), y + (h // 2)

                # 3. Find the CLOSEST Point to the subject
                distances = [np.sqrt((subj_cx - px)**2 + (subj_cy - py)**2) for px, py in POWER_POINTS]
                closest_point_idx = np.argmin(distances)
                target_x, target_y = POWER_POINTS[closest_point_idx]

                # 4. Calculate Error relative to that Power Point
                error_x = subj_cx - target_x
                error_y = subj_cy - target_y
                
                # Servo Update (Proportional)
                gain = 0.04  # Slightly lower gain for smoother composition
                if abs(error_x) > 20:
                    pan_angle -= error_x * gain
                if abs(error_y) > 20:
                    tilt_angle += error_y * gain

                # Clamp and Write
                pan_angle = max(0, min(180, pan_angle))
                tilt_angle = max(0, min(180, tilt_angle))
                kit.servo[0].angle = pan_angle
                kit.servo[1].angle = tilt_angle

                # Visual Feedback
                color = (0, 255, 255) # Cyan for tracking
                cv2.rectangle(m.array, (x, y), (x + w, y + h), color, 2)
                cv2.drawMarker(m.array, (target_x, target_y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 15, 2)
                cv2.line(m.array, (subj_cx, subj_cy), (target_x, target_y), (255, 255, 255), 1)

# Startup
config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (WIDTH, HEIGHT)},
    controls={"FrameRate": intrinsics.inference_rate}
)

picam2.pre_callback = track_rule_of_thirds
picam2.start(config, show_preview=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()