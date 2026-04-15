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
from analyzer import get_subject_sharpness, get_region_exposure, get_subject_contrast, get_region_contrast, get_frame_blur, generate_frame_metrics

class CameraWorker(QThread):
    change_pixmap_signal = pyqtSignal(QImage, QImage)
    stats_signal = pyqtSignal(dict)

    def __init__(self, model_path):
        super().__init__()
        self._run_flag = True
        self.is_tracking = False 
        self.model_path = model_path
        self.picam2 = Picamera2()
        
        # Initialize our custom hardware controller
        self.gimbal = GimbalController()
        self.active_mode = "CENTER"
        
        # --- Motion/Blur Stability Tracking ---
        self.stability_counter = 0
        self.STABILITY_REQUIRED = 2
        self.SHARP_THRESHOLD = 100.0

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
                    gray = cv2.cvtColor(main_frame, cv2.COLOR_RGB2GRAY)
                    h, w, ch = main_frame.shape
                    
                    # --- Make a copy for drawing ---
                    annotated_frame = main_frame.copy()
                    
                    # 1. Initialize these at the top of the loop
                    person_detected = False
                    current_bbox = None
                    current_comp_score = 0.0
                    
                    # 2. Hardware and YOLO Logic
                    if self.is_tracking:
                        frame_exp_val, frame_exp_status = get_region_exposure(gray)
                        
                        # --- PRIORITY 1: EXPOSURE RECOVERY ---
                        if frame_exp_status != "Good":
                            self.active_mode = f"EXPOSURE RECOVERY"
                            
                            # Slicing into regions is now only necessary to find the light source
                            regions = {
                                "top":    gray[0:h//3, :],
                                "bottom": gray[2*h//3:h, :],
                                "left":   gray[:, 0:w//3],
                                "right":  gray[:, 2*w//3:w]
                            }
                            exp_results = {name: get_region_exposure(roi) for name, roi in regions.items()}
                            
                            err_x, err_y = 0.0, 0.0
                            if exp_results['right'][1] == "Good": err_x = 0.5
                            elif exp_results['left'][1] == "Good": err_x = -0.5
                            if exp_results['top'][1] == "Good": err_y = -0.5
                            elif exp_results['bottom'][1] == "Good": err_y = 0.5
                                
                            self.gimbal.track_target(err_x, err_y)
                            cv2.putText(annotated_frame, self.active_mode, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # --- PRIORITY 2: SUBJECT TRACKING ---
                        else:
                            lores_frame = self.picam2.capture_array("lores")
                            raw_detections = hailo.run(lores_frame)
                            predictions = postproc_yolov8_pose(1, raw_detections, model_size)
                            
                            person_detected = False

                            if predictions and len(predictions['scores']) > 0:
                                best_idx = np.argmax(predictions['scores'].flatten())
                                confidence = predictions['scores'].flatten()[best_idx]

                                if confidence > 0.6:
                                    person_detected = True
                                    self.active_mode = "TRACKING (YOLO)"
                                    person_kps = predictions['keypoints'][0][best_idx].reshape(-1, 2)
                                    norm_x, norm_y = person_kps[0][0] / model_w, person_kps[0][1] / model_h
                                    
                                    # --- Image Analysis Logic ---
                                    # 1. Extract Bounding Box for the best detection (Model Space)
                                    bbox = predictions['bboxes'][0][best_idx]
                                    
                                    # 2. Scale Box to Main Frame Space (1024x768)
                                    box_x1 = int((bbox[0] / model_w) * config.WIDTH)
                                    box_y1 = int((bbox[1] / model_h) * config.HEIGHT)
                                    box_x2 = int((bbox[2] / model_w) * config.WIDTH)
                                    box_y2 = int((bbox[3] / model_h) * config.HEIGHT)
                                    
                                    current_bbox = (box_x1, box_y1, box_x2, box_y2)
                                    
                                    # 3. Draw the Sharpness Indicator Box on the ANNOTATED copy
                                    sharpness = get_subject_sharpness(main_frame, box_x1, box_y1, box_x2, box_y2)
                                    # Look at line 145 in your code:
                                    box_color = (0, 255, 0) if sharpness > 100 else (0, 0, 255)
                                    cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
                                    
                                    # Composition Logic
                                    dist_to_center = np.sqrt((norm_x - config.CENTER_TARGET[0])**2 + (norm_y - config.CENTER_TARGET[1])**2)
                                    best_rot = min(config.INTERSECTIONS, key=lambda p: np.sqrt((norm_x - p[0])**2 + (norm_y - p[1])**2))
                                    dist_to_rot = np.sqrt((norm_x - best_rot[0])**2 + (norm_y - best_rot[1])**2)

                                    # Convert the shortest distance into a 0-100 score. 
                                    # We multiply by 300 to scale the penalty (you can tweak this number to be more or less strict).
                                    best_dist = min(dist_to_center, dist_to_rot)
                                    current_comp_score = max(0.0, 100.0 - (best_dist * 300.0))
                                    
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

                                    sharpness = get_subject_sharpness(main_frame, box_x1, box_y1, box_x2, box_y2)
                                    
                                    # --- raw on annotated_frame instead of main_frame ---
                                    box_color = (0, 255, 0) if sharpness > 100 else (0, 0, 255)
                                    cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)

                                    # Visuals
                                    if self.active_mode == "CENTER":
                                        cv2.drawMarker(annotated_frame, (config.FRAME_CX, config.FRAME_CY), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)
                                    else:
                                        for px in [0.33, 0.66]:
                                            cv2.line(annotated_frame, (int(config.WIDTH * px), 0), (int(config.WIDTH * px), config.HEIGHT), (100, 100, 100), 1)
                                            cv2.line(annotated_frame, (0, int(config.HEIGHT * px)), (config.WIDTH, int(config.HEIGHT * px)), (100, 100, 100), 1)

                                    # 1. Draw Tracking Node
                                    px_x, px_y = int(norm_x * config.WIDTH), int(norm_y * config.HEIGHT)
                                    node_color = (0, 255, 0) if (abs(err_x) < config.DEADZONE and abs(err_y) < config.DEADZONE) else (0, 200, 255)
                                    cv2.circle(annotated_frame, (int(target_x * config.WIDTH), int(target_y * config.HEIGHT)), 15, (255, 255, 0), 2)
                                    cv2.circle(annotated_frame, (px_x, px_y), 8, node_color, -1)
                                    cv2.putText(annotated_frame, self.active_mode, (px_x + 15, px_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, node_color, 1)

                            # --- PRIORITY 3: IDLE ---
                            if not person_detected:
                                self.active_mode = "IDLE (NO SUBJECT)"

                    # 3. --- GENERATE AND EMIT METRICS ---
                    frame_metrics = generate_frame_metrics(
                        main_frame, gray, self.is_tracking, person_detected, current_bbox
                    )

                    # Inject our custom score here so it goes to the UI and the JSON file
                    frame_metrics["composition_score"] = round(current_comp_score, 2) 

                    self.stats_signal.emit(frame_metrics)

                    # 4. --- UI Update for BOTH frames (Add .copy() to enforce memory safety) ---
                    # Create the annotated QImage for the screen
                    rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) 
                    q_img_annotated = QImage(rgb_annotated.data, w, h, ch * w, QImage.Format_RGB888).copy()
                    
                    # Create the clean QImage for saving
                    rgb_clean = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                    q_img_clean = QImage(rgb_clean.data, w, h, ch * w, QImage.Format_RGB888).copy()

                    # Emit both!
                    self.change_pixmap_signal.emit(q_img_annotated, q_img_clean)
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Frame capture error: {e}")

    def stop(self):
        self._run_flag = False
        self.wait()