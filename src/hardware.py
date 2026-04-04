# hardware.py
import numpy as np
from adafruit_servokit import ServoKit
import config

class GimbalController:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.pan_angle = 90
        self.tilt_angle = 90
        
        # Pull servo channels from the config file
        self.pan_channel = config.PAN_CHANNEL
        self.tilt_channel = config.TILT_CHANNEL
        
        # Center the servos on startup using the configured channels
        self.kit.servo[self.pan_channel].angle = self.pan_angle
        self.kit.servo[self.tilt_channel].angle = self.tilt_angle

    def track_target(self, err_x, err_y):
        """Moves the servos based on the calculated error."""
        if abs(err_x) > config.DEADZONE:
            self.pan_angle -= (err_x * 15 * config.GAIN_PAN)
            self.pan_angle = np.clip(self.pan_angle, 0, 180)
            self.kit.servo[self.pan_channel].angle = self.pan_angle
        
        if abs(err_y) > config.DEADZONE:
            self.tilt_angle += (err_y * 15 * config.GAIN_TILT)
            self.tilt_angle = np.clip(self.tilt_angle, 0, 180)
            self.kit.servo[self.tilt_channel].angle = self.tilt_angle