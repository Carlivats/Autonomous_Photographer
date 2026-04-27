import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon

class ImageViewer(QDialog):
    """A full-screen popup to view the high-res image."""
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Full Image")
        self.resize(1024, 600) # Match your main app size
        
        # Frameless window for a slick, immersive look
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background-color: rgba(10, 12, 16, 240);") # Dark, slightly transparent background

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        
        # Give a visual cue that clicking closes it
        self.img_label.setCursor(Qt.PointingHandCursor)

        # Load the full, uncropped image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale it down so it fits nicely inside the screen without getting cut off
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.img_label.setPixmap(scaled_pixmap)

        # Allow clicking the image to close the viewer
        self.img_label.mousePressEvent = self.close_viewer

        layout.addWidget(self.img_label)

    def close_viewer(self, event):
        self.close()