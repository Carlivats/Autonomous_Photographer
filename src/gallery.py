import os
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QScrollArea, QFrame, QSizePolicy, QDialog)
from PyQt5.QtCore import Qt, QSize, QRect, QFileSystemWatcher, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap

from image_viewer import ImageViewer

class ImageThumbLabel(QLabel):
    """A custom label that loads an image, scales it, and center-crops it to fit a fixed box."""
    
    # Define a custom signal that emits the file path when clicked
    clicked = pyqtSignal(str) 

    def __init__(self, image_path, w, h):
        super().__init__()
        self.image_path = image_path
        
        self.setFixedSize(w, h)
        self.setStyleSheet("background-color: #2c3038; border-radius: 4px;")
        self.setAlignment(Qt.AlignCenter)
        
        # Change the cursor to a pointing hand so the user knows it's clickable
        self.setCursor(Qt.PointingHandCursor)

        # Load and crop the image
        original_pixmap = QPixmap(self.image_path)
        if not original_pixmap.isNull():
            target_size = QSize(w, h)
            scaled = original_pixmap.scaled(
                target_size, 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            )
            crop_rect = QRect(
                (scaled.width() - w) // 2,
                (scaled.height() - h) // 2,
                w, h
            )
            self.setPixmap(scaled.copy(crop_rect))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)

class GalleryUI(QWidget):
    # 1. Add gallery_dir=None to the parameters
    def __init__(self, parent=None, gallery_dir=None):
        super().__init__()
        self.setWindowTitle("Gallery View")
        self.resize(1024, 600)
        self.setStyleSheet("background-color: #16181d; color: white;")

        self.base_dir = os.path.dirname(__file__)
        
        # 2. Use the provided directory, or default to the local 'gallery' folder
        if gallery_dir:
            self.gallery_dir = gallery_dir
        else:
            self.gallery_dir = os.path.join(self.base_dir, 'gallery')
        
        # Ensure gallery directory exists
        os.makedirs(self.gallery_dir, exist_ok=True)

        self.setup_ui()
        self.load_images_to_grid()

        # Set up a watcher to automatically update UI when files are added/deleted
        self.watcher = QFileSystemWatcher([self.gallery_dir])
        self.watcher.directoryChanged.connect(self.load_images_to_grid)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # LEFT SIDE: Scrollable Image Grid
        # ==========================================
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.gallery_container = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_container) 
        self.gallery_layout.setSpacing(10)
        self.gallery_layout.setContentsMargins(20, 20, 20, 20)

        self.scroll_area.setWidget(self.gallery_container)
        main_layout.addWidget(self.scroll_area, stretch=3)

        # ==========================================
        # RIGHT SIDE: Sidebar Controls
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
            pixmap = QPixmap(logo_path)
            # Scale it to a reasonable width (e.g., 200px) so it doesn't blow up the sidebar
            scaled_pixmap = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_pixmap)
        else:
            # Fallback just in case the file name is wrong
            self.logo_label.setText("LOGO MISSING") 
            self.logo_label.setStyleSheet("color: red; font-weight: bold;")
            
        sidebar_layout.addWidget(self.logo_label)
        sidebar_layout.addWidget(self.create_separator_line())

        self.btn_home = self.create_sidebar_button("Home", 'home.png', is_active=True)
        self.btn_trash = self.create_sidebar_button("Trash", 'trash.png', is_active=False)
        
        sidebar_layout.addWidget(self.btn_home)
        sidebar_layout.addWidget(self.btn_trash)
        sidebar_layout.addWidget(self.create_separator_line())

        sidebar_layout.addStretch()

        self.btn_back = QPushButton(" Go Back")
        self.btn_back.setFixedHeight(50)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #1c2024;
                color: white; font-size: 18px; border-radius: 8px;
            }
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

    # --- Dynamic Grid Logic ---

    def clear_layout(self, layout):
        """Removes all widgets from the layout before a refresh."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def load_images_to_grid(self):
        """Packs images left-to-right, wrapping to a new row when space runs out."""
        self.clear_layout(self.gallery_layout)

        TARGET_HEIGHT = 180  # All images will be exactly this tall
        SPACING = 10
        # Estimate the pixel width of the left-hand scroll area
        MAX_ROW_WIDTH = 700 

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [f for f in os.listdir(self.gallery_dir) if f.lower().endswith(supported_formats)]
        
        # Sort the files by their modification time (newest first)
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.gallery_dir, x)), reverse=True)
        
        current_row_layout = QHBoxLayout()
        current_row_layout.setSpacing(SPACING)
        current_row_width = 0

        for filename in image_files:
            file_path = os.path.join(self.gallery_dir, filename)
            temp_pixmap = QPixmap(file_path)
            
            if temp_pixmap.isNull():
                continue
            
            is_landscape = temp_pixmap.width() >= temp_pixmap.height()
            
            # Mathematically perfect 16:9 and 9:16 widths based on our fixed height
            item_width = int(TARGET_HEIGHT * (16.0 / 9.0)) if is_landscape else int(TARGET_HEIGHT * (9.0 / 16.0))

            # If adding this image pushes us past the screen width, wrap to a new row!
            if current_row_width + item_width > MAX_ROW_WIDTH and current_row_width > 0:
                current_row_layout.addStretch() # Push items flush to the left
                self.gallery_layout.addLayout(current_row_layout)
                
                # Reset for the next row
                current_row_layout = QHBoxLayout()
                current_row_layout.setSpacing(SPACING)
                current_row_width = 0

            # Create the thumbnail with strict dimensions
            thumb = ImageThumbLabel(file_path, item_width, TARGET_HEIGHT)

            thumb.clicked.connect(self.open_full_image)
            
            current_row_layout.addWidget(thumb)
            
            current_row_width += (item_width + SPACING)

        # Catch the very last row and add it to the page
        if current_row_width > 0:
            current_row_layout.addStretch()
            self.gallery_layout.addLayout(current_row_layout)

        # Push all the rows to the top of the scroll area
        self.gallery_layout.addStretch()

    # --- UI Helper Methods ---

    def open_full_image(self, image_path):
        """Spawns the modal image viewer when a thumbnail is clicked."""
        self.viewer = ImageViewer(image_path, self)
        self.viewer.exec_() # exec_() halts the main UI until the popup is closed

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
            # Active State: Distinct background, bolder text
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2b3038; 
                    color: white; 
                    font-size: 18px;
                    font-weight: bold;
                    text-align: left; 
                    padding: 12px 10px; 
                    border: none;
                    border-radius: 6px;
                }
            """)
        else:
            # Idle State: Transparent background, fades out slightly
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent; 
                    color: #a0aabf; 
                    font-size: 18px;
                    text-align: left; 
                    padding: 12px 10px; 
                    border: none;
                }
                QPushButton:hover { color: white; background-color: #1c2024; border-radius: 6px;}
            """)
        return btn
    