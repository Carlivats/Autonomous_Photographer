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
        self.max_saved = 3  # The number of "Best" photos to keep
        self.top_photos = []  # List of dicts storing { 'score', 'image_path', 'json_path' }
        self.cooldown_frames = 0 
        self._debug_counter = 0
        
        # Create the gallery directory if it doesn't exist
        if not os.path.exists(self.gallery_dir):
            os.makedirs(self.gallery_dir)

    def start_session(self, target_photos=3):
        """Initializes a new continuous photography session."""
        self.max_saved = target_photos
        self.top_photos = [] # Reset our top photos cache
        self.is_active = True
        self.cooldown_frames = 0
        print(f"\n--- ?? Session Started: Hunting for the top {self.max_saved} photos ---")

    def stop_session(self):
        """Ends the active session."""
        if self.is_active:
            self.is_active = False
            print(f"--- ?? Session Complete. Saved the top {len(self.top_photos)} photos to gallery. ---")
            self.session_finished_signal.emit()

    def calculate_score(self, metrics):
        """
        Generates a composite score to rank photos against each other.
        Caps environmental metrics so texture doesn't overpower composition.
        """
        score = 0.0
        
        # 1. Cap Sharpness (Max 50 points)
        # If a face is 45, it gets 45 points. If a PC screen is 250, it gets clamped to 50.
        raw_sharpness = metrics.get("sharpness", 0)
        if isinstance(raw_sharpness, (int, float)):
            capped_sharpness = min(raw_sharpness, 50.0) 
            score += capped_sharpness
            
        # 2. Add Contrast (Bonus points, usually around 10-20, cap it just in case)
        raw_contrast = metrics.get("contrast", 0)
        if isinstance(raw_contrast, (int, float)):
            capped_contrast = min(raw_contrast, 80.0)
            score += (capped_contrast * 0.2) 
            
        # 3. Composition is the Heavyweight (0 to 100 points, multiplied by 1.5)
        # Max possible points: 150. This ensures framing is always the most important factor.
        comp_score = metrics.get("composition_score", 0)
        
        # We multiply by 1.5 to give composition a heavier weight
        if isinstance(comp_score, (int, float)):
            score += (comp_score * 1.5)
            
        return score

    def process_frame(self, metrics, q_image):
        """Called every frame. Evaluates metrics and curates the top gallery."""
        if not self.is_active:
            return

        # Cooldown gives the camera/subject time to move before analyzing again
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return

        # 1. Does it meet the minimum requirements to be a "keeper"?
        if self.should_keep_photo(metrics):
            score = self.calculate_score(metrics)

            # 2. Do we have room in our Top N list?
            if len(self.top_photos) < self.max_saved:
                self.save_and_cache_photo(metrics, q_image, score)
                self.cooldown_frames = 45 

            # 3. If full, does this new photo beat our current WORST photo?
            else:
                worst_photo = min(self.top_photos, key=lambda x: x['score'])
                
                if score > worst_photo['score']:
                    print(f"?? UPGRADE! New score ({score:.1f}) beats old score ({worst_photo['score']:.1f}). Deleting old...")
                    
                    # Delete the loser from the hard drive
                    self.delete_files(worst_photo['image_path'], worst_photo['json_path'])
                    # Remove from our tracking list
                    self.top_photos.remove(worst_photo)
                    
                    # Save the new champion
                    self.save_and_cache_photo(metrics, q_image, score)
                    self.cooldown_frames = 45 
                else:
                    # Photo was "Good", but not better than what we already have.
                    if self._debug_counter % 30 == 0:
                        print(f"Skipping: Score ({score:.1f}) wasn't high enough to break into the Top {self.max_saved}.")
                    self._debug_counter += 1

    def should_keep_photo(self, metrics):
        """The AI logic to filter out bad frames immediately."""
        reason = ""
        
        if not metrics.get("subject_info"):
            reason = "No subject detected."
        elif metrics.get("exposure_status") != "Good":
            reason = f"Bad exposure: {metrics.get('exposure_status')}"
        elif metrics.get("blur_status") != "Clear":
            reason = f"Motion blur: {metrics.get('blur_status')}"
        elif isinstance(metrics.get("sharpness"), (int, float)) and metrics.get("sharpness") < 100:
            reason = f"Subject too soft. Sharpness: {metrics.get('sharpness')}"

        if reason:
            # Print the rejection reason once every ~30 frames (about once a second)
            if getattr(self, '_debug_counter', 0) % 30 == 0:
                print(f"Skipping frame: {reason}")
            self._debug_counter = getattr(self, '_debug_counter', 0) + 1
            return False

        return True

    def save_and_cache_photo(self, metrics, q_image, score):
        """Saves the files to disk and logs them in our top_photos cache."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        basename = f"photo_{timestamp_str}"
        
        image_path = os.path.join(self.gallery_dir, f"{basename}.jpg")
        json_path = os.path.join(self.gallery_dir, f"{basename}.json")

        # Save QImage to disk
        q_image.save(image_path, "JPG", 95)

        # Save JSON metadata
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Add to our rolling cache
        self.top_photos.append({
            'score': score,
            'image_path': image_path,
            'json_path': json_path
        })
        
        print(f"?? CAPTURE SUCCESS: Saved {basename} (Score: {score:.1f}) | Gallery count: {len(self.top_photos)}/{self.max_saved}")

    def delete_files(self, img_path, json_path):
        """Safely removes the image and metadata files of discarded photos."""
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(json_path):
                os.remove(json_path)
        except Exception as e:
            print(f"Error deleting old photo files: {e}")