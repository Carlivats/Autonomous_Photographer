import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

# Resting Position
kit.servo[0].angle = 90
kit.servo[1].angle = 90

while True:
	# Ask user the servo number (1,0)
	servoNumber = int(input("Enter Servo Motor Number (1 or 0): "))
	
	# Ask the Angle (0 - 180)
	servoAngle = int(input("Enter the angle (1 to 180): "))
	
	# Move
	kit.servo[servoNumber].angle = servoAngle
