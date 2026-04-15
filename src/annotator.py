# annotator.py
import cv2
import config

class FrameAnnotator:
    @staticmethod
    def draw_tracking_ui(frame, bbox, target_x, target_y, px_x, px_y, active_mode, is_sharp, nose_px=None, box_px=None):
        """
        Draws all UI elements onto the frame.
        
        Args:
            frame: The image array to draw on (annotated_frame).
            bbox: Tuple of (x1, y1, x2, y2) for the subject bounding box.
            target_x: Normalized x coordinate of the target destination (0.0 to 1.0).
            target_y: Normalized y coordinate of the target destination (0.0 to 1.0).
            px_x: Pixel x coordinate of the subject's current tracking node.
            px_y: Pixel y coordinate of the subject's current tracking node.
            active_mode: String indicating the current tracking mode.
            is_sharp: Boolean indicating if the subject is currently sharp.
            
        Returns:
            The annotated frame.
        """
        
        # 1. Draw Bounding Box (Green if sharp, Red if soft)
        if bbox is not None:
            box_x1, box_y1, box_x2, box_y2 = bbox
            box_color = (0, 255, 0) if is_sharp else (0, 0, 255)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
        
        # 2. Draw Composition Grid / Markers
        if active_mode == "CENTER":
            # Dynamically calculate the crosshair position based on config.CENTER_TARGET
            center_px_x = int(config.CENTER_TARGET[0] * config.WIDTH)
            center_px_y = int(config.CENTER_TARGET[1] * config.HEIGHT)
            cv2.drawMarker(frame, (center_px_x, center_px_y), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)
        else:
            # Rule of Thirds Grid Lines
            # Vertical lines (33% and 66% across)
            for px in [0.33, 0.66]:
                x_pos = int(config.WIDTH * px)
                cv2.line(frame, (x_pos, 0), (x_pos, config.HEIGHT), (100, 100, 100), 1)
            
            # Horizontal lines (33% and 66% down)
            for py in [0.33, 0.66]:
                y_pos = int(config.HEIGHT * py)
                cv2.line(frame, (0, y_pos), (config.WIDTH, y_pos), (100, 100, 100), 1)

        # 3. Draw Sensor Fusion Visualization (The Twist)
        if nose_px and box_px:
            # Draw a thin white line connecting the Box Anchor to the Nose Anchor
            cv2.line(frame, nose_px, box_px, (200, 200, 200), 1)
            
            # Draw the Nose Point (Magenta)
            cv2.circle(frame, nose_px, 4, (255, 0, 255), -1)
            
            # Draw the Box Point (Cyan)
            cv2.circle(frame, box_px, 4, (255, 255, 0), -1)
                
        # 4. Draw Tracking Nodes
        target_px_x = int(target_x * config.WIDTH)
        target_px_y = int(target_y * config.HEIGHT)
        cv2.circle(frame, (target_px_x, target_px_y), 15, (255, 255, 0), 2)
        
        # Draw the Definite Point (The weighted average)
        node_color = (0, 255, 0) if is_sharp else (0, 200, 255) 
        cv2.circle(frame, (px_x, px_y), 8, node_color, -1)
        
        # Draw the active mode text right next to the subject node
        cv2.putText(frame, active_mode, (px_x + 15, px_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, node_color, 1)
        
        return frame