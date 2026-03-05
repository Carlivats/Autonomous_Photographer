import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2
from adafruit_servokit import ServoKit

# Setup Servos
kit = ServoKit(channels=16)
pan_angle, tilt_angle = 90, 90
kit.servo[0].angle = pan_angle
kit.servo[1].angle = tilt_angle

# Setup Camera
WIDTH, HEIGHT = 640, 480
FRAME_CX, FRAME_CY = WIDTH // 2, HEIGHT // 2

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": 'RGB888', "size": (WIDTH, HEIGHT)})
picam2.configure(config)
picam2.start()

# 3. Tracking Constants
GAIN = 0.04 
DEADZONE = 20

try:
    while True:
        frame = picam2.capture_array()

        # AI Processing (Downscale for speed)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame)

        if face_locations:
            # Get the first face
            top, right, bottom, left = face_locations[0]
            
            # Scale back up
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            face_x = (left + right) // 2
            face_y = (top + bottom) // 2

            # --- MOTOR LOGIC ---
            error_x = face_x - FRAME_CX
            error_y = face_y - FRAME_CY

            # Only move if error is bigger than the deadzone
            if abs(error_x) > DEADZONE:
                # Subtracting because if face is to the right, we move the servo angle left (or vice versa)
                pan_angle -= error_x * GAIN
            
            if abs(error_y) > DEADZONE:
                # Tilt is usually inverted depending on your mount
                tilt_angle += error_y * GAIN

            # Keep angles within safe bounds (0-180)
            pan_angle = np.clip(pan_angle, 0, 180)
            tilt_angle = np.clip(tilt_angle, 0, 180)

            # Update Servos
            kit.servo[0].angle = pan_angle
            kit.servo[1].angle = tilt_angle

            # Drawing
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.circle(frame, (face_x, face_y), 5, (0, 255, 0), -1)

        # Draw Center Target
        cv2.drawMarker(frame, (FRAME_CX, FRAME_CY), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

        cv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Autonomous Tracking', cv_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    picam2.stop()
    cv2.destroyAllWindows()