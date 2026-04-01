import cv2
import numpy as np

class CenteredSymmetry:
    def __init__(self, gain_pan=0.5, gain_tilt=0.15, deadzone=0.03):
        """
        Initializes the Centered Symmetry behavior.
        GAIN: Lower = Smoother, Higher = Snappier
        DEADZONE: Percentage of the screen center that is considered "safe"
        """
        self.gain_pan = gain_pan
        self.gain_tilt = gain_tilt
        self.deadzone = deadzone

    def process(self, predictions, model_w, model_h, frame):
        """
        Analyzes the frame and predictions, draws overlays, and returns steering commands.
        
        Returns:
            pan_delta (float): The amount to adjust the pan angle.
            tilt_delta (float): The amount to adjust the tilt angle.
            status (str): The current tracking status.
        """
        pan_delta = 0.0
        tilt_delta = 0.0
        status = "SEARCHING"

        # 1. Draw Static Center Guide on the frame
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)

        if predictions and 'scores' in predictions:
            scores = predictions['scores']
            keypoints = predictions['keypoints']

            if len(scores) > 0:
                # 2. Get the most confident person
                best_idx = np.argmax(scores.flatten())
                confidence = scores.flatten()[best_idx]

                if confidence > 0.6: 
                    person_kps = keypoints[best_idx].reshape(-1, 2)
                    
                    # 3. Get Normalized Coordinates
                    norm_x = person_kps[0][0] / model_w
                    norm_y = person_kps[0][1] / model_h
                    
                    # 4. Calculate Error relative to center (0.5, 0.5)
                    err_x = norm_x - 0.5
                    err_y = norm_y - 0.5

                    # 5. Calculate Steering Logic (Deltas)
                    if abs(err_x) > self.deadzone:
                        # Negative because camera moves opposite to pixel direction
                        pan_delta = -(err_x * 15 * self.gain_pan)
                    
                    if abs(err_y) > self.deadzone:
                        tilt_delta = (err_y * 15 * self.gain_tilt)

                    status = "STABLE" if (pan_delta == 0 and tilt_delta == 0) else "MOVING"

                    # 6. Draw Tracking Visuals
                    px_x, px_y = int(norm_x * w), int(norm_y * h)
                    color = (0, 255, 0) if status == "STABLE" else (0, 200, 255)
                    cv2.circle(frame, (px_x, px_y), 8, color, -1)
                    cv2.putText(frame, status, (px_x + 15, px_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return pan_delta, tilt_delta, status