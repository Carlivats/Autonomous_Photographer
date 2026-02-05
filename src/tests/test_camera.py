from picamera2 import Picamera2, Preview
import os
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration({"size": (1280, 720)})
picam2.configure(config)

picam2.start_preview(Preview.QTGL)
picam2.start()

time.sleep(5)

if not os.path.exists("images"):
    os.makedirs("images")

picam2.capture_file("images/test4.jpg")

# Stop the camera
picam2.stop()