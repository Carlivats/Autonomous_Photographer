# main.py
import sys
import argparse
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# Import our custom worker thread
from vision import CameraWorker

class AutonomousPhotographerUI(QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("Autonomous Photographer")
        self.resize(1024, 600)
        self.show()
        self.setStyleSheet("background-color: #16181d;")

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Video Feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #1c2024;")
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(320, 240)
        main_layout.addWidget(self.video_label, stretch=3)

        # Control Panel
        control_layout = QVBoxLayout()
        button_style = """
            QPushButton {
                background-color: #1c2024; color: white;
                font-size: 20px; font-weight: bold;
                padding: 15px; border-radius: 8px;
            }
            QPushButton:pressed { background-color: #525a70; }
        """

        self.btn_start = QPushButton("Start Session", self)
        self.btn_start.setStyleSheet(button_style.replace("#1c2024", "#fed766").replace("white", "black"))
        self.btn_start.clicked.connect(self.start_session)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop Session", self)
        self.btn_stop.setStyleSheet(button_style)
        self.btn_stop.clicked.connect(self.stop_session)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        control_layout.addStretch()

        self.btn_exit = QPushButton("Exit App", self)
        self.btn_exit.setStyleSheet(button_style.replace("#1c2024", "#e74c3c"))
        self.btn_exit.clicked.connect(self.close_app)
        control_layout.addWidget(self.btn_exit)

        main_layout.addLayout(control_layout, stretch=1)

        # Start Background Thread
        self.thread = CameraWorker(model_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, q_img):
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_session(self):
        self.thread.is_tracking = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_session(self):
        self.thread.is_tracking = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

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