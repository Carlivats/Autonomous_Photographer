# composition.py
import numpy as np
import config

class CompositionEngine:
    @staticmethod
    def get_best_target(norm_x, norm_y, current_mode):
        """Calculates the best composition target and returns the (x,y), mode, and distance."""
        # 1. Calculate distances to center and closest Rule-of-Thirds intersection
        dist_to_center = np.sqrt((norm_x - config.CENTER_TARGET[0])**2 + (norm_y - config.CENTER_TARGET[1])**2)
        best_rot = min(config.INTERSECTIONS, key=lambda p: np.sqrt((norm_x - p[0])**2 + (norm_y - p[1])**2))
        dist_to_rot = np.sqrt((norm_x - best_rot[0])**2 + (norm_y - best_rot[1])**2)
        
        # 2. Apply configuration biases (prevents the camera from jittering between points)
        biased_dist_to_rot = dist_to_rot * config.THIRDS_BIAS 
        if current_mode == "THIRDS":
            biased_dist_to_rot *= config.STICKINESS 

        # 3. Determine the winner
        if dist_to_center < biased_dist_to_rot:
            return config.CENTER_TARGET, "CENTER", dist_to_center
        else:
            return best_rot, "THIRDS", dist_to_rot

    @staticmethod
    def calculate_framing_score(best_dist):
        """Converts distance-to-target into a 0-100 score."""
        return max(0.0, 100.0 - (best_dist * 300.0))