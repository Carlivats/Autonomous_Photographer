import sys
import time
import cv2
import numpy as np
import argparse
from adafruit_servokit import ServoKit
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo 

# Note: Ensure pose_utils.py is in the same folder as this script
from pose_utils import postproc_yolov8_pose 

# --- Setup Arguments & Model Path ---
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef')
args = parser.parse_args()

# --- Servo Setup ---
kit = ServoKit(channels=16)
pan_angle = 90
tilt_angle = 90
kit.servo[0].angle = pan_angle
kit.servo[1].angle = tilt_angle

# --- Constants ---
# Using the resolution from Hailo example
WIDTH, HEIGHT = 1024, 768
FRAME_CX, FRAME_CY = WIDTH // 2, HEIGHT // 2
GAIN_PAN = 0.5   # Lower = Smoother, Higher = Snappier
GAIN_TILT = 0.15  
DEADZONE = 0.03   # 3% of the screen center is "safe"
last_predictions = None

def draw_and_track(request):
    global pan_angle, tilt_angle
    
    with MappedArray(request, 'main') as m:
        # Draw Center Guide
        cv2.drawMarker(m.array, (FRAME_CX, FRAME_CY), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)
        
        if last_predictions:
            try:
                scores = last_predictions['scores']
                keypoints = last_predictions['keypoints']

                if len(scores) > 0:
                    # 1. Get the most confident person
                    best_idx = np.argmax(scores.flatten())
                    confidence = scores.flatten()[best_idx]

                    if confidence > 0.6: # Higher threshold for movement stability
                        person_kps = keypoints[best_idx].reshape(-1, 2)
                        
                        # 2. Get Normalized Coordinates (0.0 to 1.0)
                        # Hailo returns coords relative to model_size (model_w, model_h)
                        norm_x = person_kps[0][0] / model_w
                        norm_y = person_kps[0][1] / model_h
                        
                        # 3. Calculate Error relative to center (0.5, 0.5)
                        err_x = norm_x - 0.5
                        err_y = norm_y - 0.5

                        # 4. Smooth Servo Logic
                        # Only move if error is outside the 3% deadzone
                        if abs(err_x) > DEADZONE:
                            # We subtract for pan because camera usually moves 
                            # opposite to pixel direction for centering
                            pan_angle -= (err_x * 15 * GAIN_PAN) # Scale error to degrees
                            pan_angle = np.clip(pan_angle, 0, 180)
                            kit.servo[0].angle = pan_angle
                        
                        if abs(err_y) > DEADZONE:
                            tilt_angle += (err_y * 15 * GAIN_TILT)
                            tilt_angle = np.clip(tilt_angle, 0, 180)
                            kit.servo[1].angle = tilt_angle

                        # 5. Visuals (scaled for your 1024x768 display)
                        px_x, px_y = int(norm_x * WIDTH), int(norm_y * HEIGHT)
                        color = (0, 255, 0) if (abs(err_x) < DEADZONE and abs(err_y) < DEADZONE) else (0, 200, 255)
                        cv2.circle(m.array, (px_x, px_y), 8, color, -1)
                        cv2.putText(m.array, "STABLE" if color == (0, 255, 0) else "MOVING", 
                                    (px_x + 15, px_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                pass

# --- Main Logic ---
# Ssame as example
with Hailo(args.model) as hailo:
    model_h, model_w, _ = hailo.get_input_shape()
    model_size = (model_w, model_h)

    with Picamera2() as picam2:
        config = picam2.create_video_configuration(
            main={'size': (WIDTH, HEIGHT), 'format': 'XRGB8888'},
            lores={'size': model_size, 'format': 'RGB888'}
        )
        picam2.configure(config)
        picam2.start_preview(Preview.QTGL, x=0, y=0, width=WIDTH, height=HEIGHT)
        picam2.start()
        picam2.pre_callback = draw_and_track

        try:
            while True:
                frame = picam2.capture_array('lores')
                raw_detections = hailo.run(frame)
                last_predictions = postproc_yolov8_pose(1, raw_detections, model_size)
        except KeyboardInterrupt:
            print("\nExiting...")