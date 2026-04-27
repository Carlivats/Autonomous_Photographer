# config.py

# Camera & Display Settings
WIDTH = 1024
HEIGHT = 768
FRAME_CX = WIDTH // 2
FRAME_CY = HEIGHT // 2

# Tracking Constants
GAIN_PAN = 0.5
GAIN_TILT = 0.15
DEADZONE = 0.03
STICKINESS = 0.8
CENTER_WIDTH = 0.25

# 75% power to the nose (accuracy) / 25% power to the box (stability)
NOSE_WEIGHT = 0.50
BOX_WEIGHT = 0.50

# Vertical position for Centered Symmetry (0.0 to 1.0)
# 0.5 is dead center (lots of headroom). 0.33 is the upper third line (standard portrait).
CENTER_FACE_Y = 0.33 

# X is exactly middle (0.5), Y uses our portrait offset
CENTER_TARGET = (0.5, CENTER_FACE_Y)
INTERSECTIONS = [
    (0.33, 0.33), (0.66, 0.33),
    (0.33, 0.66), (0.66, 0.66)
]

# Servo Channels
PAN_CHANNEL = 1
TILT_CHANNEL = 0

# Analysis Thresholds
SHARPNESS_THRESHOLD = 60.0

MAX_SAVED_PHOTOS = 5