# composition.py
import numpy as np
import config

class CompositionEngine:
    @staticmethod
    def get_best_target(norm_x, norm_y, current_mode):
        """Calculates the best composition target based on screen columns."""
        
        # 1. Define the basic boundaries of the central column (33% to 66%)
        half_width = config.CENTER_WIDTH / 2.0
        left_bound = 0.5 - half_width
        right_bound = 0.5 + half_width
        
        # 2. Hysteresis / Stickiness
        # We add a 5% buffer so the camera doesn't violently jitter 
        # back and forth if the subject is standing exactly on the line.
        buffer = 0.05 
        
        if current_mode == "CENTER":
            # Make the center column wider to "hold onto" the subject
            left_bound -= buffer  
            right_bound += buffer 
        elif current_mode == "THIRDS":
            # Make the center column narrower so they have to commit to moving middle
            left_bound += buffer  
            right_bound -= buffer 

        # 3. Determine the mode based on horizontal position
        if left_bound <= norm_x <= right_bound:
            # Subject is in the middle column
            best_target = config.CENTER_TARGET
            new_mode = "CENTER"
        else:
            # Subject is in the left or right outer columns.
            best_target = min(config.INTERSECTIONS, key=lambda p: np.sqrt((norm_x - p[0])**2 + (norm_y - p[1])**2))
            new_mode = "THIRDS"

        # 4. Calculate the distance to the chosen target
        distance = np.sqrt((norm_x - best_target[0])**2 + (norm_y - best_target[1])**2)

        return best_target, new_mode, distance

    @staticmethod
    def calculate_framing_score(best_dist):
        """Converts distance-to-target into a 0-100 score."""
        return max(0.0, 100.0 - (best_dist * 300.0))