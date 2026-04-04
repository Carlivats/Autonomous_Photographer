# main.py
import os
import sys
import argparse
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon
from vision import CameraWorker
from session_manager import CaptureSessionManager

class AutonomousPhotographerUI(QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("Autonomous Photographer")
        self.resize(1024, 600)
        self.show()
        self.setStyleSheet("background-color: #16181d;")

        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'gear.png')

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

        control_layout.addSpacing(20) 
        
        self.btn_settings = QPushButton("", self)
        self.btn_settings.setFixedSize(80, 80)
        
        self.btn_settings.setIcon(QIcon(icon_path))
        
        # Scale the image inside the button (40x40 leaves nice padding inside the 80x80 button)
        self.btn_settings.setIconSize(QSize(75, 75)) 
        
        self.btn_settings.setStyleSheet("""
            QPushButton {
                background-color: #1c2024; 
                border-radius: 40px; 
                border: 2px solid #525a70;
            }
            QPushButton:hover { border: 2px solid white; background-color: #2b3038; }
            QPushButton:pressed { background-color: #525a70; }
        """)
        self.btn_settings.clicked.connect(self.open_settings)

        # Center it horizontally
        settings_layout = QHBoxLayout()
        settings_layout.addStretch()
        settings_layout.addWidget(self.btn_settings)
        settings_layout.addStretch()
        
        control_layout.addLayout(settings_layout)

        self.stats_label = QLabel(self.video_label)

        control_layout.addStretch()
        
        # 1. Define the dynamic styles for the button
        self.btn_idle_style = """
            QPushButton {
                background-color: white; 
                border-radius: 40px;  /* Exactly half the width/height to make a circle */
                border: 4px solid #525a70;
            }
            QPushButton:pressed { background-color: #d0d0d0; }
        """
        
        self.btn_active_style = """
            QPushButton {
                background-color: #e74c3c; /* Red */
                border-radius: 40px;
                border: 4px solid white;
            }
            QPushButton:pressed { background-color: #c0392b; }
        """

        # 2. Create the toggle button
        self.btn_toggle = QPushButton("", self)
        self.btn_toggle.setFixedSize(80, 80)
        self.btn_toggle.setStyleSheet(self.btn_idle_style)
        self.btn_toggle.clicked.connect(self.toggle_session)

        # 3. Center the button horizontally using an internal layout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_toggle)
        btn_layout.addStretch()

        # Add the centered button to the main control panel
        control_layout.addLayout(btn_layout)
        
        control_layout.addStretch()
        
        # Use RGBA for the background color to set the alpha channel (transparency)
        # 166 out of 255 is roughly 65% opacity (35% transparent)
        self.stats_label.setStyleSheet("""
            background-color: rgba(28, 32, 36, 166); 
            color: #a0aabf; 
            font-size: 16px; 
            font-family: monospace;
            padding: 15px;
            border-radius: 0px;
        """)

        # Initialize with placeholder text
        self.stats_label.setText("Sharpness : N/A\nExposure  : N/A\nContrast  : N/A\nScene Blur: N/A")
        
        # 2. Position it in the top-left corner with a 15px margin
        self.stats_label.move(15, 15)
        
        # 3. Ensure it sizes itself correctly right on startup
        self.stats_label.adjustSize()

        self.btn_gallery = QPushButton("Gallery", self)
        self.btn_gallery.setStyleSheet(button_style)
        self.btn_gallery.clicked.connect(self.open_gallery)
        control_layout.addWidget(self.btn_gallery)

        self.btn_exit = QPushButton("Exit App", self)
        self.btn_exit.setStyleSheet(button_style.replace("#1c2024", "#e74c3c").replace("#525a70", "#c0392b"))
        self.btn_exit.clicked.connect(self.close_app)
        control_layout.addWidget(self.btn_exit)

        main_layout.addLayout(control_layout, stretch=1)

        # --- Initialize Session Manager ---
        self.session_manager = CaptureSessionManager()
        self.session_manager.session_finished_signal.connect(self.end_session_ui)

        # --- Initialize and Start the Camera Thread ---
        self.thread = CameraWorker(model_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats_panel)
        
        self.thread.start()

    # --- UI Slots (Actions) ---
    def update_image(self, q_img):
        # Save the current frame to memory so the session manager can grab it
        self.current_frame = q_img 
        
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def toggle_session(self):
        """Toggles the autonomous tracking state and updates the button color."""
        if not self.thread.is_tracking:
            # Turn ON
            print("Starting autonomous session...")
            self.thread.is_tracking = True
            self.btn_toggle.setStyleSheet(self.btn_active_style)
            
            self.session_manager.start_session(target_photos=3)
        else:
            # Turn OFF Manually
            self.end_session_ui()

    def end_session_ui(self):
        """Resets the UI when a session ends automatically or manually."""
        print("Stopping autonomous session...")
        self.thread.is_tracking = False
        self.session_manager.stop_session()
        self.btn_toggle.setStyleSheet(self.btn_idle_style)

    def update_stats_panel(self, stats):
        """Updates the text label whenever the vision thread emits new data."""
        sharpness = stats.get("sharpness", "N/A")
        exposure = stats.get("exposure", "N/A")
        exposure_status = stats.get("exposure_status", "N/A")
        contrast = stats.get("contrast", "N/A")
        contrast_status = stats.get("contrast_status", "N/A")
        blur = stats.get("blur", "N/A")
        blur_status = stats.get("blur_status", "N/A")
        
        display_text = (
            f"Sharpness : {sharpness}\n"
            f"Exposure  : {exposure} ({exposure_status})\n"
            f"Contrast  : {contrast} ({contrast_status})\n"
            f"Scene Blur: {blur} ({blur_status})"
        )
        self.stats_label.setText(display_text)
        
        # Add this line to force the transparent box to snap to the new text size
        self.stats_label.adjustSize()
        
        if self.thread.is_tracking and hasattr(self, 'current_frame'):
            self.session_manager.process_frame(stats, self.current_frame)

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