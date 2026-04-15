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
from composition import CompositionEngine
from annotator import FrameAnnotator

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
                    
                    annotated_frame = main_frame.copy()
                    
                    # 1. Initialize per-frame variables
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

                            if predictions and len(predictions['scores']) > 0:
                                best_idx = np.argmax(predictions['scores'].flatten())
                                confidence = predictions['scores'].flatten()[best_idx]

                                if confidence > 0.6:
                                    person_detected = True
                                    
                                    # 2A. Extract spatial data
                                    bbox = predictions['bboxes'][0][best_idx]
                                    person_kps = predictions['keypoints'][0][best_idx].reshape(-1, 2)
                                    
                                    # --- POINT 1: The Exact Keypoint (Nose) ---
                                    nose_norm_x = person_kps[0][0] / model_w
                                    nose_norm_y = person_kps[0][1] / model_h
                                    
                                    # --- POINT 2: The Calculated Bounding Box Center-Top ---
                                    box_w = bbox[2] - bbox[0]
                                    box_h = bbox[3] - bbox[1]
                                    
                                    # Center of the X axis
                                    box_norm_x = (bbox[0] + (box_w / 2.0)) / model_w
                                    
                                    # 15% down from the top edge (Usually where the center of the face sits in a YOLO box)
                                    box_norm_y = (bbox[1] + (box_h * 0.15)) / model_h 
                                    
                                    # --- THE TWIST: Create the Definite Point ---
                                    
                                    norm_x = (nose_norm_x * config.NOSE_WEIGHT) + (box_norm_x * config.BOX_WEIGHT)
                                    norm_y = (nose_norm_y * config.NOSE_WEIGHT) + (box_norm_y * config.BOX_WEIGHT)

                                    # Calculate bounding box for the drawing/sharpness logic
                                    box_x1 = int((bbox[0] / model_w) * config.WIDTH)
                                    box_y1 = int((bbox[1] / model_h) * config.HEIGHT)
                                    box_x2 = int((bbox[2] / model_w) * config.WIDTH)
                                    box_y2 = int((bbox[3] / model_h) * config.HEIGHT)
                                    current_bbox = (box_x1, box_y1, box_x2, box_y2)
                                    
                                    # Convert all three points to pixels
                                    # The Definite Point
                                    px_x = int(norm_x * config.WIDTH)
                                    px_y = int(norm_y * config.HEIGHT)
                                    
                                    # The Nose Point
                                    nose_px = (int(nose_norm_x * config.WIDTH), int(nose_norm_y * config.HEIGHT))
                                    
                                    # The Box Point
                                    box_px = (int(box_norm_x * config.WIDTH), int(box_norm_y * config.HEIGHT))
                                    
                                    # 2B. Composition Math
                                    target_pos, self.active_mode, best_dist = CompositionEngine.get_best_target(norm_x, norm_y, self.active_mode)
                                    current_comp_score = CompositionEngine.calculate_framing_score(best_dist)

                                    # 2C. Hardware Movement
                                    err_x = norm_x - target_pos[0]
                                    err_y = norm_y - target_pos[1]
                                    self.gimbal.track_target(err_x, err_y)

                                    # 2D. Image Analysis
                                    sharpness = get_subject_sharpness(main_frame, box_x1, box_y1, box_x2, box_y2)
                                    is_sharp = sharpness > self.SHARP_THRESHOLD
                                    
                                    # 2E. Drawing
                                    annotated_frame = FrameAnnotator.draw_tracking_ui(
                                        annotated_frame, current_bbox, target_pos[0], target_pos[1], 
                                        px_x, px_y, self.active_mode, is_sharp, nose_px, box_px
                                    )

                            # --- PRIORITY 3: IDLE ---
                            if not person_detected:
                                self.active_mode = "IDLE (NO SUBJECT)"

                    # 3. --- GENERATE AND EMIT METRICS ---
                    frame_metrics = generate_frame_metrics(
                        main_frame, gray, self.is_tracking, person_detected, current_bbox
                    )
                    
                    # Inject our new composition score here
                    frame_metrics["composition_score"] = round(current_comp_score, 2)
                    
                    self.stats_signal.emit(frame_metrics)

                    # 4. --- UI Update for BOTH frames ---
                    rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) 
                    q_img_annotated = QImage(rgb_annotated.data, w, h, ch * w, QImage.Format_RGB888).copy()
                    
                    rgb_clean = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
                    q_img_clean = QImage(rgb_clean.data, w, h, ch * w, QImage.Format_RGB888).copy()

                    self.change_pixmap_signal.emit(q_img_annotated, q_img_clean)
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Frame capture error: {e}")

    def stop(self):
        self._run_flag = False
        self.wait()