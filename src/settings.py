import os
import re
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QFrame, QSlider, QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap

# Import the config to read current values
import config

class SettingsUI(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("Settings View")
        self.resize(1024, 600)
        self.setStyleSheet("background-color: #16181d; color: white;")

        self.base_dir = os.path.dirname(__file__)
        self.config_path = os.path.join(self.base_dir, 'config.py')
        
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # LEFT SIDE: Settings Controls
        # ==========================================
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        settings_layout.setSpacing(30)

        title_label = QLabel("Camera & Tracking Settings")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffffff;")
        settings_layout.addWidget(title_label)

        # --- Setting 1: Sharpness Threshold ---
        self.lbl_sharpness = QLabel(f"Sharpness Threshold: {config.SHARPNESS_THRESHOLD}")
        self.lbl_sharpness.setStyleSheet("font-size: 18px;")
        
        self.slider_sharpness = QSlider(Qt.Horizontal)
        self.slider_sharpness.setRange(10, 150)
        self.slider_sharpness.setValue(int(config.SHARPNESS_THRESHOLD))
        self.slider_sharpness.valueChanged.connect(
            lambda v: self.lbl_sharpness.setText(f"Sharpness Threshold: {v}")
        )
        
        settings_layout.addWidget(self.lbl_sharpness)
        settings_layout.addWidget(self.slider_sharpness)

        # --- Setting 2: Max Saved Photos ---
        max_photos_layout = QHBoxLayout()
        lbl_max_photos = QLabel("Max Saved Photos:")
        lbl_max_photos.setStyleSheet("font-size: 18px;")
        
        self.spin_max_photos = QSpinBox()
        self.spin_max_photos.setRange(1, 100)
        self.spin_max_photos.setValue(config.MAX_SAVED_PHOTOS)
        self.spin_max_photos.setStyleSheet("""
            QSpinBox { background-color: #2b3038; color: white; font-size: 18px; padding: 5px; border: 1px solid #525a70; }
        """)
        
        max_photos_layout.addWidget(lbl_max_photos)
        max_photos_layout.addWidget(self.spin_max_photos)
        max_photos_layout.addStretch()
        settings_layout.addLayout(max_photos_layout)

        # --- Setting 3: Tracking Focus (Box vs Nose Weight combined) ---
        # 0 = 100% Box, 100 = 100% Nose
        current_nose_pct = int(config.NOSE_WEIGHT * 100)
        self.lbl_weight = QLabel(f"Tracking Focus (Stability vs Accuracy): {current_nose_pct}% Nose")
        self.lbl_weight.setStyleSheet("font-size: 18px;")
        
        self.slider_weight = QSlider(Qt.Horizontal)
        self.slider_weight.setRange(0, 100)
        self.slider_weight.setValue(current_nose_pct)
        self.slider_weight.valueChanged.connect(self.update_weight_label)

        # Helper labels for the slider
        weight_labels_layout = QHBoxLayout()
        lbl_box = QLabel("More Box (Stable)")
        lbl_box.setStyleSheet("color: #a0aabf; font-size: 14px;")
        lbl_nose = QLabel("More Nose (Accurate)")
        lbl_nose.setStyleSheet("color: #a0aabf; font-size: 14px;")
        lbl_nose.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        weight_labels_layout.addWidget(lbl_box)
        weight_labels_layout.addWidget(lbl_nose)

        settings_layout.addWidget(self.lbl_weight)
        settings_layout.addWidget(self.slider_weight)
        settings_layout.addLayout(weight_labels_layout)

        settings_layout.addStretch()

        # Save Button
        self.btn_save = QPushButton("Save & Apply Settings")
        self.btn_save.setFixedHeight(60)
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; font-size: 20px; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:pressed { background-color: #1e8449; }
        """)
        self.btn_save.clicked.connect(self.save_settings)
        settings_layout.addWidget(self.btn_save)

        main_layout.addWidget(settings_container, stretch=3)

        # ==========================================
        # RIGHT SIDE: Sidebar Controls (Matches Gallery)
        # ==========================================
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color: #121418;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(30, 40, 30, 40)
        sidebar_layout.setSpacing(15)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(self.base_dir, 'assets', 'logo.png') 
        
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaledToWidth(200, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
        else:
            self.logo_label.setText("LOGO MISSING") 
            self.logo_label.setStyleSheet("color: red; font-weight: bold;")
            
        sidebar_layout.addWidget(self.logo_label)
        sidebar_layout.addWidget(self.create_separator_line())

        self.btn_active_tab = self.create_sidebar_button("Settings", 'gear.png', is_active=True)
        sidebar_layout.addWidget(self.btn_active_tab)
        sidebar_layout.addWidget(self.create_separator_line())

        sidebar_layout.addStretch()

        self.btn_back = QPushButton(" Go Back")
        self.btn_back.setFixedHeight(50)
        self.btn_back.setStyleSheet("""
            QPushButton { background-color: #1c2024; color: white; font-size: 18px; border-radius: 8px; }
            QPushButton:hover { background-color: #2b3038; }
            QPushButton:pressed { background-color: #525a70; }
        """)
        back_icon_path = os.path.join(self.base_dir, 'assets', 'back.png')
        if os.path.exists(back_icon_path):
            self.btn_back.setIcon(QIcon(back_icon_path))
            self.btn_back.setIconSize(QSize(24, 24))
            
        self.btn_back.clicked.connect(self.close) 
        sidebar_layout.addWidget(self.btn_back)

        main_layout.addWidget(sidebar, stretch=1)

    def update_weight_label(self, value):
        self.lbl_weight.setText(f"Tracking Focus (Stability vs Accuracy): {value}% Nose")

    def create_separator_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #4a505c; max-height: 1px;")
        return line

    def create_sidebar_button(self, text, icon_filename, is_active=False):
        btn = QPushButton(f" {text}")
        icon_path = os.path.join(self.base_dir, 'assets', icon_filename)
        if os.path.exists(icon_path):
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(24, 24))
            
        if is_active:
            btn.setStyleSheet("""
                QPushButton { background-color: #2b3038; color: white; font-size: 18px; font-weight: bold; text-align: left; padding: 12px 10px; border: none; border-radius: 6px; }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton { background-color: transparent; color: #a0aabf; font-size: 18px; text-align: left; padding: 12px 10px; border: none; }
                QPushButton:hover { color: white; background-color: #1c2024; border-radius: 6px;}
            """)
        return btn

    def save_settings(self):
        """Updates the live config and rewrites the config.py file for persistence."""
        new_sharpness = float(self.slider_sharpness.value())
        new_max_photos = self.spin_max_photos.value()
        
        new_nose_weight = self.slider_weight.value() / 100.0
        new_box_weight = 1.0 - new_nose_weight

        # 1. Update active memory so it works immediately without restarting
        config.SHARPNESS_THRESHOLD = new_sharpness
        config.MAX_SAVED_PHOTOS = new_max_photos
        config.NOSE_WEIGHT = new_nose_weight
        config.BOX_WEIGHT = new_box_weight

        # 2. Rewrite the config.py file so settings persist after reboot
        try:
            with open(self.config_path, 'r') as file:
                file_contents = file.read()

            # Use Regex to find and replace the variable lines
            file_contents = re.sub(r'SHARPNESS_THRESHOLD\s*=\s*[\d.]+', f'SHARPNESS_THRESHOLD = {new_sharpness}', file_contents)
            file_contents = re.sub(r'MAX_SAVED_PHOTOS\s*=\s*\d+', f'MAX_SAVED_PHOTOS = {new_max_photos}', file_contents)
            file_contents = re.sub(r'NOSE_WEIGHT\s*=\s*[\d.]+', f'NOSE_WEIGHT = {new_nose_weight:.2f}', file_contents)
            file_contents = re.sub(r'BOX_WEIGHT\s*=\s*[\d.]+', f'BOX_WEIGHT = {new_box_weight:.2f}', file_contents)

            with open(self.config_path, 'w') as file:
                file.write(file_contents)

            # Show success message
            QMessageBox.information(self, "Success", "Settings saved successfully!")
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings to config.py:\n{str(e)}")