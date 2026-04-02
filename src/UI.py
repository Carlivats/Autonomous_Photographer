import sys
import time
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from picamera2 import Picamera2


class CameraWorker(QThread):
    # This signal will carry the QImage from the background thread to the GUI
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.picam2 = Picamera2()

    def run(self):
        # Configure the camera
        config = self.picam2.create_video_configuration(
            {"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Continuous capture loop in the background
        while self._run_flag:
            try:
                frame = self.picam2.capture_array("main")
                
                # --- ADD THIS LINE TO FIX THE COLORS ---
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                self.change_pixmap_signal.emit(q_img)
                time.sleep(0.03) 
            except Exception as e:
                print(f"Frame capture error: {e}")

        self.picam2.stop()

    def stop(self):
        self._run_flag = False
        self.wait()


class AutonomousPhotographerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autonomous Photographer")
        
        # Set a default window size (Width, Height)
        self.resize(1024, 600) 
        # Show as a standard window instead of full screen
        self.show() 
        
        self.setStyleSheet("background-color: #16181d;")

        # --- Main Horizontal Layout ---
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # --- Left Side: Video Feed ---
        # --- Left Side: Video Feed ---
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #1c2024;")
        
        # --- ADD THESE TWO LINES ---
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(320, 240) # Gives it a safe baseline so it doesn't collapse
        # ---------------------------

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
        self.thread = CameraWorker()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    # --- UI Slots (Actions) ---
    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def start_session(self):
        print("Starting autonomous session...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_session(self):
        print("Stopping autonomous session...")
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
    app = QApplication(sys.argv)
    window = AutonomousPhotographerUI()
    sys.exit(app.exec_())