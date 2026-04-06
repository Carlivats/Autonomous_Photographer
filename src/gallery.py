import os
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QScrollArea, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

class AspectRatioLabel(QLabel):
    """A custom label that strictly maintains either a 16:9 or 9:16 aspect ratio."""
    def __init__(self, is_landscape=True, has_content=False):
        super().__init__()
        self.is_landscape = is_landscape
        
        # Sizing policy that tells Qt this widget relies on height-for-width calculations
        policy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        
        # Prevent the boxes from collapsing too small
        self.setMinimumWidth(120)
        
        if has_content:
            self.setStyleSheet("background-color: #4a505c; border-radius: 4px;") 
        else:
            self.setStyleSheet("background-color: #2c3038; border-radius: 4px;")

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        # Calculate strict 16:9 or 9:16 ratios
        if self.is_landscape:
            return int(width * (9.0 / 16.0))
        else:
            return int(width * (16.0 / 9.0))
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Force the height constraint on resize so the grid doesn't squash them
        w = self.width()
        h = self.heightForWidth(w)
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)

class GalleryUI(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("Gallery View")
        self.resize(1024, 600)
        self.setStyleSheet("background-color: #16181d; color: white;")

        # Base directory for loading icons
        self.base_dir = os.path.dirname(__file__)

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

        gallery_container = QWidget()
        gallery_layout = QGridLayout(gallery_container)
        gallery_layout.setSpacing(10)
        gallery_layout.setContentsMargins(20, 20, 20, 20)
        
        # Row 0: Three equal 16:9 landscape blocks
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=True), 0, 0, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=True), 0, 2, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=True), 0, 4, 1, 2)
        
        # Row 1: Staggered 16:9 (wide) and 9:16 (narrow) blocks
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=True), 1, 0, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=False, has_content=False), 1, 2, 1, 1)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=True), 1, 3, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=False, has_content=False), 1, 5, 1, 1)

        # Row 2: Standard 16:9 blocks
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=False), 2, 0, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=False), 2, 2, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=False), 2, 4, 1, 2)
        
        # Row 3: 9:16 and 16:9 blocks at the bottom
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=False, has_content=False), 3, 0, 1, 1)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=False), 3, 1, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=True, has_content=False), 3, 3, 1, 2)
        gallery_layout.addWidget(AspectRatioLabel(is_landscape=False, has_content=False), 3, 5, 1, 1)

        # Push elements to the top if the window is tall
        gallery_layout.setRowStretch(gallery_layout.rowCount(), 1)

        self.scroll_area.setWidget(gallery_container)
        main_layout.addWidget(self.scroll_area, stretch=3)

        # ==========================================
        # RIGHT SIDE: Sidebar Controls
        # ==========================================
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color: #121418;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(30, 40, 30, 40)
        sidebar_layout.setSpacing(15)

        # Title
        title_label = QLabel("Autonomous\nPhotographer Logo")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        sidebar_layout.addWidget(title_label)

        # Separator Line
        sidebar_layout.addWidget(self.create_separator_line())

        # Menu Buttons (Now using imported QIcons)
        self.btn_home = self.create_sidebar_button("Home", 'home.png')
        self.btn_trash = self.create_sidebar_button("Trash", 'trash.png')
        sidebar_layout.addWidget(self.btn_home)
        sidebar_layout.addWidget(self.btn_trash)

        # Separator Line
        sidebar_layout.addWidget(self.create_separator_line())

        # Push the bottom button down
        sidebar_layout.addStretch()

        # Go Back Button
        self.btn_back = QPushButton(" Go Back")
        self.btn_back.setFixedHeight(50)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #1c2024;
                color: white;
                font-size: 18px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #2b3038; }
            QPushButton:pressed { background-color: #525a70; }
        """)
        
        # Load Back Icon
        back_icon_path = os.path.join(self.base_dir, 'assets', 'back.png')
        self.btn_back.setIcon(QIcon(back_icon_path))
        self.btn_back.setIconSize(QSize(24, 24))
        
        # Connect to close the gallery window
        self.btn_back.clicked.connect(self.close) 
        sidebar_layout.addWidget(self.btn_back)

        main_layout.addWidget(sidebar, stretch=1)

    # --- UI Helper Methods ---

    def create_separator_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #4a505c; max-height: 1px;")
        return line

    def create_sidebar_button(self, text, icon_filename):
        btn = QPushButton(f" {text}") # Small space before text
        
        # Build path and set icon
        icon_path = os.path.join(self.base_dir, 'assets', icon_filename)
        btn.setIcon(QIcon(icon_path))
        btn.setIconSize(QSize(24, 24))
        
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #d1d5db;
                font-size: 18px;
                text-align: left;
                padding: 12px 10px;
                border: none;
            }
            QPushButton:hover { color: white; background-color: #1c2024; border-radius: 6px;}
        """)
        return btn