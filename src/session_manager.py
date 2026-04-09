# session_manager.py
import os
import json
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal

class CaptureSessionManager(QObject):
    # This signal will tell main.py when the session is over
    session_finished_signal = pyqtSignal()

    def __init__(self, gallery_dir="gallery"):
        super().__init__()
        self.gallery_dir = gallery_dir
        self.is_active = False
        self.target_photos = 0
        self.photos_taken = 0
        self.cooldown_frames = 0 # Prevents taking 5 photos in a fraction of a second
        
        # Create the gallery directory if it doesn't exist
        if not os.path.exists(self.gallery_dir):
            os.makedirs(self.gallery_dir)

    def start_session(self, target_photos=3):
        """Initializes a new photography session."""
        self.target_photos = target_photos
        self.photos_taken = 0
        self.is_active = True
        self.cooldown_frames = 0
        print(f"--- Session Started: Aiming for {target_photos} high-quality photos ---")

    def stop_session(self):
        """Ends the active session."""
        if self.is_active:
            self.is_active = False
            print("--- Session Complete ---")
            self.session_finished_signal.emit()

    def process_frame(self, metrics, q_image):
        """Called every frame. Evaluates metrics and captures the photo if perfect."""
        if not self.is_active:
            return

        # Wait a bit after taking a photo to let the gimbal settle or the subject move
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return

        # The Brain: Decide whether to keep or discard this frame
        if self.should_keep_photo(metrics):
            self.save_photo(metrics, q_image)
            self.photos_taken += 1
            self.cooldown_frames = 45 # roughly 1.5 seconds cooldown at ~30fps

            if self.photos_taken >= self.target_photos:
                self.stop_session()

    def should_keep_photo(self, metrics):
        """The AI logic to evaluate frame quality with debug reporting."""
        reason = ""
        
        # 1. Must have a subject detected
        if not metrics.get("subject_info"):
            reason = "No subject detected."
            
        # 2. Exposure must be balanced
        elif metrics.get("exposure_status") != "Good":
            reason = f"Bad exposure: {metrics.get('exposure_status')}"
            
        # 3. Scene must be stable
        elif metrics.get("blur_status") != "Clear":
            reason = f"Motion blur: {metrics.get('blur_status')}"
            
        # 4. Subject must be in focus
        elif isinstance(metrics.get("sharpness"), (int, float)) and metrics.get("sharpness") < 100:
            reason = f"Subject too soft. Sharpness: {metrics.get('sharpness')}"

        # If there is a reason to reject, print it occasionally and return False
        if reason:
            # Print the rejection reason once every ~30 frames (about once a second)
            if getattr(self, '_debug_counter', 0) % 30 == 0:
                print(f"Skipping frame: {reason}")
            self._debug_counter = getattr(self, '_debug_counter', 0) + 1
            return False

        # If it passes all tests, it's a keeper!
        return True

    def save_photo(self, metrics, q_image):
        """Saves the image and JSON metadata to the gallery."""
        # Generate a unique basename using the timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        basename = f"photo_{timestamp_str}"
        
        image_path = os.path.join(self.gallery_dir, f"{basename}.jpg")
        json_path = os.path.join(self.gallery_dir, f"{basename}.json")

        # Save QImage to disk
        q_image.save(image_path, "JPG", 95)

        # Save JSON metadata
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"CAPTURE SUCCESS: Saved {basename}")