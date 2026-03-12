import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
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
        config = self.picam2.create_video_configuration(
            {"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Continuous capture loop in the background
        while self._run_flag:
            try:
                # Grab the frame array
                frame = self.picam2.capture_array("main")
                
                # Convert the NumPy array to a PyQt QImage
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Emit the signal to send the image to the main GUI
                self.change_pixmap_signal.emit(q_img)
                
                # Small sleep to yield resources and hit roughly 30-60 FPS
                time.sleep(0.03) 
            except Exception as e:
                print(f"Frame capture error: {e}")

        # Clean up when the thread stops
        self.picam2.stop()

    def stop(self):
        # Gracefully break the while loop
        self._run_flag = False
        self.wait()


class AutonomousPhotographerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autonomous Photographer")
        
        # Make the window fullscreen for the Pi touchscreen
        self.showFullScreen()
        self.setStyleSheet("background-color: black;")

        # Create a layout to stack widgets vertically
        layout = QVBoxLayout()

        # Create the label that will hold the video frames
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Create a touch-friendly exit button
        self.exit_btn = QPushButton("Exit Preview", self)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
            }
        """)
        self.exit_btn.clicked.connect(self.close_app)
        layout.addWidget(self.exit_btn)

        self.setLayout(layout)

        # --- Initialize and Start the Camera Thread ---
        self.thread = CameraWorker()
        # Connect the thread's signal to our update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, q_img):
        """This slot receives the QImage and displays it."""
        # Convert QImage to QPixmap and set it on the label
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def close_app(self):
        """Safely shut down the worker thread before closing."""
        self.thread.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutonomousPhotographerUI()
    sys.exit(app.exec_())