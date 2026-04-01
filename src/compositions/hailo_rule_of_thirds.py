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
INTERSECTIONS = [
    (0.33, 0.33), (0.66, 0.33),
    (0.33, 0.66), (0.66, 0.66)
]

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
                        
                        # 2. Get Normalized Coordinates
                        norm_x = person_kps[0][0] / model_w
                        norm_y = person_kps[0][1] / model_h

                        # 3. Find the closest Rule of Thirds intersection
                        # We find the intersection point with the smallest Euclidean distance to the nose
                        best_target = min(INTERSECTIONS, key=lambda p: np.sqrt((norm_x - p[0])**2 + (norm_y - p[1])**2))
                        target_x, target_y = best_target

                        # Calculate Error relative to the chosen intersection
                        err_x = norm_x - target_x
                        err_y = norm_y - target_y

                        # 4. Smooth Servo Logic (Same as before, but using new err_x/y)
                        if abs(err_x) > DEADZONE:
                            pan_angle -= (err_x * 15 * GAIN_PAN)
                            pan_angle = np.clip(pan_angle, 0, 180)
                            kit.servo[0].angle = pan_angle
                        
                        if abs(err_y) > DEADZONE:
                            tilt_angle += (err_y * 15 * GAIN_TILT)
                            tilt_angle = np.clip(tilt_angle, 0, 180)
                            kit.servo[1].angle = tilt_angle

                        # 5. Visuals: Draw the Grid and the Target
                        # Draw vertical lines
                        cv2.line(m.array, (int(WIDTH * 0.33), 0), (int(WIDTH * 0.33), HEIGHT), (100, 100, 100), 1)
                        cv2.line(m.array, (int(WIDTH * 0.66), 0), (int(WIDTH * 0.66), HEIGHT), (100, 100, 100), 1)
                        # Draw horizontal lines
                        cv2.line(m.array, (0, int(HEIGHT * 0.33)), (WIDTH, int(HEIGHT * 0.33)), (100, 100, 100), 1)
                        cv2.line(m.array, (0, int(HEIGHT * 0.66)), (WIDTH, int(HEIGHT * 0.66)), (100, 100, 100), 1)
                        
                        # Highlight the active target intersection
                        cv2.circle(m.array, (int(target_x * WIDTH), int(target_y * HEIGHT)), 15, (255, 255, 0), 2)
            except Exception as e:
                pass

# --- Main Logic ---
# Same as example
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