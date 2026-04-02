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
THIRDS_BIAS = 0.4
STICKINESS = 0.8

# Composition Targets
CENTER_TARGET = (0.5, 0.5)
INTERSECTIONS = [
    (0.33, 0.33), (0.66, 0.33),
    (0.33, 0.66), (0.66, 0.66)
]