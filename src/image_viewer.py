import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon

class ImageViewer(QDialog):
    """A full-screen popup to view the high-res image."""
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path # Save the path so we can delete it later
        
        self.setWindowTitle("Full Image")
        self.resize(1024, 600) 
        
        # Frameless window for a slick, immersive look
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background-color: rgba(10, 12, 16, 240);") 

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setCursor(Qt.PointingHandCursor)

        # Load the full image
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.img_label.setPixmap(scaled_pixmap)

        # Allow clicking the background/image to close the viewer
        self.img_label.mousePressEvent = self.close_viewer
        layout.addWidget(self.img_label)

        # ==========================================
        # FLOATING TRASH BUTTON
        # ==========================================
        self.btn_trash = QPushButton(" Move to Trash", self)
        self.btn_trash.setFixedSize(160, 50)
        
        # Absolute position: Top right corner (X=1024-160-30, Y=30)
        self.btn_trash.move(834, 30) 
        
        self.btn_trash.setStyleSheet("""
            QPushButton {
                background-color: rgba(231, 76, 60, 210); /* Semi-transparent Red */
                color: white; 
                font-size: 18px; 
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: rgba(231, 76, 60, 255); border: 2px solid white; }
            QPushButton:pressed { background-color: #c0392b; }
        """)
        
        # Optional: Add the trash icon if it exists
        trash_icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'trash.png')
        if os.path.exists(trash_icon_path):
            self.btn_trash.setIcon(QIcon(trash_icon_path))
            self.btn_trash.setIconSize(QSize(24, 24))
            
        self.btn_trash.clicked.connect(self.delete_image)
        
        # Force the button to stay on top of the image label
        self.btn_trash.raise_()

    def delete_image(self):
        """Asks for confirmation, deletes the image and JSON, and closes the viewer."""
        # Ask the user if they are sure (optional, remove this block for 1-click delete)
        confirm = QMessageBox.question(
            self, "Confirm Delete", 
            "Are you sure you want to delete this photo?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # 1. Delete the image file
                if os.path.exists(self.image_path):
                    os.remove(self.image_path)
                
                # 2. Delete the associated JSON data file
                # This replaces '.jpg' or '.png' with '.json'
                json_path = self.image_path.rsplit('.', 1)[0] + '.json'
                if os.path.exists(json_path):
                    os.remove(json_path)
                    
                print(f"Deleted: {os.path.basename(self.image_path)}")
                
                # 3. Close the modal (the FileSystemWatcher will handle the Gallery refresh!)
                self.close()
                
            except Exception as e:
                print(f"Error deleting files: {e}")

    def close_viewer(self, event):
        self.close()