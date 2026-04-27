import os
import shutil
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon

class ImageViewer(QDialog):
    """A full-screen popup to view the high-res image."""
    def __init__(self, image_path, is_trash=False, parent=None):
        super().__init__(parent)
        self.image_path = image_path 
        self.is_trash = is_trash # Tracks if we are viewing a trashed image
        
        self.setWindowTitle("Full Image")
        self.resize(1024, 600) 
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background-color: rgba(10, 12, 16, 240);") 

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setCursor(Qt.PointingHandCursor)

        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled_pixmap)

        self.img_label.mousePressEvent = self.close_viewer
        layout.addWidget(self.img_label)

        # ==========================================
        # CONTEXTUAL FLOATING BUTTONS
        # ==========================================
        if not self.is_trash:
            # HOME MODE: Move to Trash Button
            self.btn_trash = QPushButton(" Move to Trash", self)
            self.btn_trash.setFixedSize(160, 50)
            self.btn_trash.move(834, 30) 
            self.btn_trash.setStyleSheet("""
                QPushButton { background-color: rgba(231, 76, 60, 210); color: white; font-size: 18px; font-weight: bold; border-radius: 8px; }
                QPushButton:hover { background-color: rgba(231, 76, 60, 255); border: 2px solid white; }
            """)
            self.btn_trash.clicked.connect(self.move_to_trash)
            self.btn_trash.raise_()
            
        else:
            # TRASH MODE: Restore Button
            self.btn_restore = QPushButton(" Restore", self)
            self.btn_restore.setFixedSize(130, 50)
            self.btn_restore.move(684, 30) # Shifted to the left to make room
            self.btn_restore.setStyleSheet("""
                QPushButton { background-color: rgba(39, 174, 96, 210); color: white; font-size: 18px; font-weight: bold; border-radius: 8px; }
                QPushButton:hover { background-color: rgba(39, 174, 96, 255); border: 2px solid white; }
            """)
            self.btn_restore.clicked.connect(self.restore_image)
            self.btn_restore.raise_()

            # TRASH MODE: Delete Forever Button
            self.btn_delete = QPushButton(" Delete Forever", self)
            self.btn_delete.setFixedSize(160, 50)
            self.btn_delete.move(834, 30) 
            self.btn_delete.setStyleSheet("""
                QPushButton { background-color: rgba(192, 57, 43, 210); color: white; font-size: 18px; font-weight: bold; border-radius: 8px; }
                QPushButton:hover { background-color: rgba(192, 57, 43, 255); border: 2px solid white; }
            """)
            self.btn_delete.clicked.connect(self.delete_permanently)
            self.btn_delete.raise_()

    def get_json_path(self, img_path):
        """Helper to find the matching JSON file."""
        return img_path.rsplit('.', 1)[0] + '.json'

    def move_to_trash(self):
        """Moves the image and JSON to the .trash folder."""
        current_dir = os.path.dirname(self.image_path)
        trash_dir = os.path.join(current_dir, '.trash')
        os.makedirs(trash_dir, exist_ok=True)
        
        try:
            shutil.move(self.image_path, os.path.join(trash_dir, os.path.basename(self.image_path)))
            json_path = self.get_json_path(self.image_path)
            if os.path.exists(json_path):
                shutil.move(json_path, os.path.join(trash_dir, os.path.basename(json_path)))
            self.close()
        except Exception as e:
            print(f"Error moving to trash: {e}")

    def restore_image(self):
        """Moves the image and JSON back to the main gallery folder."""
        trash_dir = os.path.dirname(self.image_path)
        home_dir = os.path.dirname(trash_dir) # Go up one level
        
        try:
            shutil.move(self.image_path, os.path.join(home_dir, os.path.basename(self.image_path)))
            json_path = self.get_json_path(self.image_path)
            if os.path.exists(json_path):
                shutil.move(json_path, os.path.join(home_dir, os.path.basename(json_path)))
            self.close()
        except Exception as e:
            print(f"Error restoring image: {e}")

    def delete_permanently(self):
        """Permanently erases the files from the hard drive."""
        confirm = QMessageBox.question(self, "Warning", "Permanently delete this photo?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            try:
                if os.path.exists(self.image_path): os.remove(self.image_path)
                json_path = self.get_json_path(self.image_path)
                if os.path.exists(json_path): os.remove(json_path)
                self.close()
            except Exception as e:
                print(f"Error deleting files: {e}")

    def close_viewer(self, event):
        self.close()