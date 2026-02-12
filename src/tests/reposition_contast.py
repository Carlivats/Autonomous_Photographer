import time
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
from adafruit_servokit import ServoKit
from src.get_contrast_score import get_contrast_score

model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
model = IMX500(model_path)
picam2 = Picamera2(model.camera_num)

# Setup Servos
kit = ServoKit(channels=16)
pan_angle = 90
tilt_angle = 90
kit.servo[0].angle = pan_angle
kit.servo[1].angle = tilt_angle

config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
picam2.start(config, show_preview=True)

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Divide into regions
        regions = {
            "top":    gray[0:h//3, :],
            "bottom": gray[2*h//3:h, :],
            "left":   gray[:, 0:w//3],
            "right":  gray[:, 2*w//3:w],
            "center": gray[h//3:2*h//3, w//3:2*w//3]
        }

        # Calculate contrast for all
        results = {name: get_contrast_score(roi) for name, roi in regions.items()}
        
        # Decision Logic: Seek HIGHER contrast
        step = 2
        if results['right'][0] > results['center'][0] + 3:
            pan_angle = max(0, pan_angle - step)
        elif results['left'][0] > results['center'][0] + 3:
            pan_angle = min(180, pan_angle + step)

        if results['top'][0] > results['center'][0] + 3:
            tilt_angle = max(0, tilt_angle - step)
        elif results['bottom'][0] > results['center'][0] + 3:
            tilt_angle = min(180, tilt_angle + step)

        kit.servo[0].angle = pan_angle
        kit.servo[1].angle = tilt_angle
        
        print(f"Center Contrast: {results['center'][0]:.1f} ({results['center'][1]}) | P:{pan_angle} T:{tilt_angle}")
        time.sleep(0.1)

except KeyboardInterrupt:
    picam2.stop()