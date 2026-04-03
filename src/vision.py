# vision.py
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from picamera2 import Picamera2
from picamera2.devices import Hailo

import config
from hardware import GimbalController
from pose_utils import postproc_yolov8_pose 
from analyzer import get_subject_sharpness

class CameraWorker(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, model_path):
        super().__init__()
        self._run_flag = True
        self.is_tracking = False 
        self.model_path = model_path
        self.picam2 = Picamera2()
        
        # Initialize our custom hardware controller
        self.gimbal = GimbalController()
        self.active_mode = "CENTER"

    def run(self):
        with Hailo(self.model_path) as hailo:
            model_h, model_w, _ = hailo.get_input_shape()
            model_size = (model_w, model_h)

            cam_config = self.picam2.create_video_configuration(
                main={"size": (config.WIDTH, config.HEIGHT), "format": "RGB888"},
                lores={"size": model_size, "format": "RGB888"}
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            
            while self._run_flag:
                try:
                    main_frame = self.picam2.capture_array("main")
                    
                    if self.is_tracking:
                        lores_frame = self.picam2.capture_array("lores")
                        raw_detections = hailo.run(lores_frame)
                        # Note: predictions['bboxes'] contains boxes in [x1, y1, x2, y2]
                        predictions = postproc_yolov8_pose(1, raw_detections, model_size)

                        if predictions and len(predictions['scores']) > 0:
                            best_idx = np.argmax(predictions['scores'].flatten())
                            confidence = predictions['scores'].flatten()[best_idx]

                            if confidence > 0.6:
                                person_kps = predictions['keypoints'][0][best_idx].reshape(-1, 2)
                                norm_x = person_kps[0][0] / model_w
                                norm_y = person_kps[0][1] / model_h
                                
                                # --- NEW: Image Analysis Logic ---
                                # 1. Extract Bounding Box for the best detection (Model Space)
                                bbox = predictions['bboxes'][0][best_idx]
                                
                                # 2. Scale Box to Main Frame Space (1024x768)
                                box_x1 = int((bbox[0] / model_w) * config.WIDTH)
                                box_y1 = int((bbox[1] / model_h) * config.HEIGHT)
                                box_x2 = int((bbox[2] / model_w) * config.WIDTH)
                                box_y2 = int((bbox[3] / model_h) * config.HEIGHT)
                                
                                # 3. Calculate Sharpness on the high-res main_frame
                                sharpness = get_subject_sharpness(main_frame, box_x1, box_y1, box_x2, box_y2)
                                
                                # 4. Draw the bounding box and sharpness score
                                box_color = (0, 255, 0) if sharpness > 100 else (0, 0, 255)
                                cv2.rectangle(main_frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
                                cv2.putText(main_frame, f"Sharp: {int(sharpness)}", (box_x1, max(20, box_y1 - 10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                                # ---------------------------------
                                
                                # Composition Logic
                                dist_to_center = np.sqrt((norm_x - config.CENTER_TARGET[0])**2 + (norm_y - config.CENTER_TARGET[1])**2)
                                best_rot = min(config.INTERSECTIONS, key=lambda p: np.sqrt((norm_x - p[0])**2 + (norm_y - p[1])**2))
                                dist_to_rot = np.sqrt((norm_x - best_rot[0])**2 + (norm_y - best_rot[1])**2)
                                
                                biased_dist_to_rot = dist_to_rot * config.THIRDS_BIAS 
                                if self.active_mode == "THIRDS":
                                    biased_dist_to_rot *= config.STICKINESS 

                                if dist_to_center < biased_dist_to_rot:
                                    target_x, target_y = config.CENTER_TARGET
                                    self.active_mode = "CENTER"
                                else:
                                    target_x, target_y = best_rot
                                    self.active_mode = "THIRDS"

                                # Hardware Movement
                                err_x, err_y = norm_x - target_x, norm_y - target_y
                                self.gimbal.track_target(err_x, err_y)

                                # Visuals
                                if self.active_mode == "CENTER":
                                    cv2.drawMarker(main_frame, (config.FRAME_CX, config.FRAME_CY), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)
                                else:
                                    for px in [0.33, 0.66]:
                                        cv2.line(main_frame, (int(config.WIDTH * px), 0), (int(config.WIDTH * px), config.HEIGHT), (100, 100, 100), 1)
                                        cv2.line(main_frame, (0, int(config.HEIGHT * px)), (config.WIDTH, int(config.HEIGHT * px)), (100, 100, 100), 1)

                                # 1. Draw Tracking Node
                                px_x, px_y = int(norm_x * config.WIDTH), int(norm_y * config.HEIGHT)
                                node_color = (0, 255, 0) if (abs(err_x) < config.DEADZONE and abs(err_y) < config.DEADZONE) else (0, 200, 255)
                                cv2.circle(main_frame, (int(target_x * config.WIDTH), int(target_y * config.HEIGHT)), 15, (255, 255, 0), 2)
                                cv2.circle(main_frame, (px_x, px_y), 8, node_color, -1)
                                cv2.putText(main_frame, self.active_mode, (px_x + 15, px_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, node_color, 1)
                                         
                    # UI Update
                    h, w, ch = main_frame.shape
                    q_img = QImage(main_frame.data, w, h, ch * w, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(q_img)
                    time.sleep(0.01) 
                    
                except Exception as e:
                    print(f"Frame capture error: {e}")

            self.picam2.stop()

    def stop(self):
        self._run_flag = False
        self.wait()