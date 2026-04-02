import sys
import time
import cv2
import numpy as np
import argparse
from adafruit_servokit import ServoKit

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from picamera2 import Picamera2
from picamera2.devices import Hailo 
from pose_utils import postproc_yolov8_pose


class CameraWorker(QThread):
    # This signal will carry the QImage from the background thread to the GUI
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, model_path):
        super().__init__()
        self._run_flag = True
        self.is_tracking = False  # --- NEW: Toggle for tracking state ---
        self.model_path = model_path
        self.picam2 = Picamera2()

        # --- Servo Setup ---
        self.kit = ServoKit(channels=16)
        self.pan_angle = 90
        self.tilt_angle = 90
        self.kit.servo[0].angle = self.pan_angle
        self.kit.servo[1].angle = self.tilt_angle

        # --- Tracking Constants ---
        self.WIDTH, self.HEIGHT = 1024, 768
        self.FRAME_CX, self.FRAME_CY = self.WIDTH // 2, self.HEIGHT // 2
        self.GAIN_PAN = 0.5
        self.GAIN_TILT = 0.15  
        self.DEADZONE = 0.03

    def run(self):
        # Initialize Hailo inside the thread
        with Hailo(self.model_path) as hailo:
            model_h, model_w, _ = hailo.get_input_shape()
            model_size = (model_w, model_h)

            # Configure the camera with both main and lores streams
            config = self.picam2.create_video_configuration(
                main={"size": (self.WIDTH, self.HEIGHT), "format": "RGB888"},
                lores={"size": model_size, "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()

            # Continuous capture loop
            while self._run_flag:
                try:
                    # Capture the high-res frame for the UI
                    main_frame = self.picam2.capture_array("main")
                    
                    # --- AI TRACKING LOGIC ---
                    if self.is_tracking:
                        # Draw Center Guide
                        cv2.drawMarker(main_frame, (self.FRAME_CX, self.FRAME_CY), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)

                        # Capture low-res frame and run inference
                        lores_frame = self.picam2.capture_array("lores")
                        raw_detections = hailo.run(lores_frame)
                        predictions = postproc_yolov8_pose(1, raw_detections, model_size)

                        if predictions and len(predictions['scores']) > 0:
                            scores = predictions['scores']
                            keypoints = predictions['keypoints']

                            # 1. Get the most confident person
                            best_idx = np.argmax(scores.flatten())
                            confidence = scores.flatten()[best_idx]

                            if confidence > 0.6:
                                person_kps = keypoints[best_idx].reshape(-1, 2)
                                
                                # 2. Get Normalized Coordinates
                                norm_x = person_kps[0][0] / model_w
                                norm_y = person_kps[0][1] / model_h
                                
                                # 3. Calculate Error relative to center
                                err_x = norm_x - 0.5
                                err_y = norm_y - 0.5

                                # 4. Smooth Servo Logic
                                if abs(err_x) > self.DEADZONE:
                                    self.pan_angle -= (err_x * 15 * self.GAIN_PAN)
                                    self.pan_angle = np.clip(self.pan_angle, 0, 180)
                                    self.kit.servo[0].angle = self.pan_angle
                                
                                if abs(err_y) > self.DEADZONE:
                                    self.tilt_angle += (err_y * 15 * self.GAIN_TILT)
                                    self.tilt_angle = np.clip(self.tilt_angle, 0, 180)
                                    self.kit.servo[1].angle = self.tilt_angle

                                # 5. Visuals
                                px_x, px_y = int(norm_x * self.WIDTH), int(norm_y * self.HEIGHT)
                                color = (0, 255, 0) if (abs(err_x) < self.DEADZONE and abs(err_y) < self.DEADZONE) else (0, 200, 255)
                                cv2.circle(main_frame, (px_x, px_y), 8, color, -1)
                                cv2.putText(main_frame, "STABLE" if color == (0, 255, 0) else "MOVING", 
                                            (px_x + 15, px_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # --- UI UPDATE LOGIC ---
                    h, w, ch = main_frame.shape
                    bytes_per_line = ch * w
                    
                    # Ensure format matches standard RGB UI output
                    q_img = QImage(main_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(q_img)
                    
                    time.sleep(0.01) 
                    
                except Exception as e:
                    print(f"Frame capture error: {e}")

            self.picam2.stop()
            
    def stop(self):
        self._run_flag = False
        self.wait()


class AutonomousPhotographerUI(QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("Autonomous Photographer")
        
        self.resize(1024, 600)
        self.show()
        self.setStyleSheet("background-color: #16181d;")

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # --- Left Side: Video Feed ---
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #1c2024;")
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(320, 240)
        main_layout.addWidget(self.video_label, stretch=3)

        # --- Right Side: Control Panel ---
        control_layout = QVBoxLayout()
        
        button_style = """
            QPushButton {
                background-color: #1c2024;
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
            }
            QPushButton:pressed {
                background-color: #525a70;
            }
        """

        self.btn_start = QPushButton("Start Session", self)
        self.btn_start.setStyleSheet(button_style.replace("#1c2024", "#fed766").replace("#525a70", "#fed766").replace("white", "black"))
        self.btn_start.clicked.connect(self.start_session)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop Session", self)
        self.btn_stop.setStyleSheet(button_style)
        self.btn_stop.clicked.connect(self.stop_session)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        control_layout.addStretch()

        self.btn_settings = QPushButton("Settings", self)
        self.btn_settings.setStyleSheet(button_style)
        self.btn_settings.clicked.connect(self.open_settings)
        control_layout.addWidget(self.btn_settings)

        self.btn_gallery = QPushButton("Gallery", self)
        self.btn_gallery.setStyleSheet(button_style)
        self.btn_gallery.clicked.connect(self.open_gallery)
        control_layout.addWidget(self.btn_gallery)

        self.btn_exit = QPushButton("Exit App", self)
        self.btn_exit.setStyleSheet(button_style.replace("#1c2024", "#e74c3c").replace("#525a70", "#c0392b"))
        self.btn_exit.clicked.connect(self.close_app)
        control_layout.addWidget(self.btn_exit)

        main_layout.addLayout(control_layout, stretch=1)

        # --- Initialize and Start the Camera Thread ---
        self.thread = CameraWorker(model_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    # --- UI Slots (Actions) ---
    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def start_session(self):
        print("Starting autonomous session...")
        self.thread.is_tracking = True # ENABLE TRACKING
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_session(self):
        print("Stopping autonomous session...")
        self.thread.is_tracking = False # DISABLE TRACKING
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def open_settings(self):
        print("Opening Settings...")

    def open_gallery(self):
        print("Opening Gallery...")

    def close_app(self):
        self.thread.stop()
        self.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = AutonomousPhotographerUI(args.model)
    sys.exit(app.exec_())