# analyzer.py
import time
import cv2

def generate_frame_metrics(main_frame, gray_frame, is_tracking, person_detected=False, bbox=None):
    """
    Aggregates all image quality metrics into a single dictionary.
    This fulfills the metadata requirement for Benchmark 7.
    """
    # Always-on metrics
    frame_exp_val, frame_exp_status = get_region_exposure(gray_frame)
    frame_blur_val, frame_blur_status = get_frame_blur(gray_frame)
    
    # Default empty/N/A values for subject-specific metrics
    sharpness = "N/A"
    contrast_val = "N/A"
    contrast_status = "N/A"
    composition_score = 0.0 # Placeholder for future composition rating engine
    subject_info = None

    if is_tracking and person_detected and bbox is not None:
        # Unpack bounding box
        box_x1, box_y1, box_x2, box_y2 = bbox
        
        # Calculate subject-specific metrics
        sharpness = int(get_subject_sharpness(main_frame, box_x1, box_y1, box_x2, box_y2))
        contrast_val, contrast_status = get_subject_contrast(main_frame, box_x1, box_y1, box_x2, box_y2)
        
        # Format subject info for JSON storage
        subject_info = {
            "type": "person",
            "bbox": [box_x1, box_y1, box_x2, box_y2]
        }
    else:
        # If no subject, calculate room contrast
        room_cont_val, room_cont_status = get_region_contrast(gray_frame)
        contrast_val = int(room_cont_val)
        contrast_status = room_cont_status

    # Assemble the final dictionary matching Benchmark 7 requirements
    metrics = {
        "timestamp": time.time(),
        "sharpness": sharpness,
        "exposure": int(frame_exp_val),
        "exposure_status": frame_exp_status,
        "contrast": contrast_val,
        "contrast_status": contrast_status,
        "blur": int(frame_blur_val),
        "blur_status": frame_blur_status,
        "composition_score": composition_score,
        "subject_info": subject_info
    }
    
    return metrics

def get_subject_sharpness(image, x1, y1, x2, y2):
    """
    Calculates the Laplacian variance (sharpness) for a specific bounding box.
    Expects coordinates in (x1, y1, x2, y2) format.
    """
    img_h, img_w = image.shape[:2]
    
    # 1. Boundary Protection (ensure coordinates don't exceed image dimensions)
    x_start, y_start = max(0, int(x1)), max(0, int(y1))
    x_end, y_end = min(img_w, int(x2)), min(img_h, int(y2))
    
    # 2. Crop the image to the bounding box
    subject_roi = image[y_start:y_end, x_start:x_end]
    if subject_roi.size == 0:
        return 0.0
        
    # 3. Convert crop to grayscale
    gray_subject = cv2.cvtColor(subject_roi, cv2.COLOR_BGR2GRAY)
    
    # 4. Calculate Sharpness Score
    sharpness_score = cv2.Laplacian(gray_subject, cv2.CV_64F).var()
    
    return sharpness_score

# Placeholders
def get_subject_contrast(image, x1, y1, x2, y2):
    pass

def get_region_exposure(roi):
    """
    Analyzes the brightness levels of a raw image array (Region of Interest).
    Returns the average brightness and an exposure status.
    """
    mean_brightness, _ = cv2.meanStdDev(roi)
    exposure = mean_brightness[0][0]

    if exposure < 50:
        status = "Underexposed"
    elif exposure > 160:
        status = "Overexposed"
    else:
        status = "Good"

    return exposure, status

def get_subject_contrast(image, x1, y1, x2, y2):
    """
    Calculates the RMS contrast for the tracked subject's bounding box.
    """
    img_h, img_w = image.shape[:2]
    
    x_start, y_start = max(0, int(x1)), max(0, int(y1))
    x_end, y_end = min(img_w, int(x2)), min(img_h, int(y2))
    
    subject_roi = image[y_start:y_end, x_start:x_end]
    if subject_roi.size == 0:
        return 0.0, "N/A"

    gray_subject = cv2.cvtColor(subject_roi, cv2.COLOR_BGR2GRAY)
    _, contrast = cv2.meanStdDev(gray_subject)
    contrast_val = contrast[0][0]

    if contrast_val < 30: status = "Very Low"
    elif contrast_val < 50: status = "Low/Med"
    elif contrast_val < 80: status = "Good"
    else: status = "High"
    
    return contrast_val, status

def get_region_contrast(roi):
    """
    Calculates the RMS contrast of a raw image array.
    """
    _, contrast = cv2.meanStdDev(roi)
    contrast_val = contrast[0][0]
    
    if contrast_val < 30: status = "Very Low"
    elif contrast_val < 50: status = "Low/Med"
    elif contrast_val < 80: status = "Good"
    else: status = "High"
    
    return contrast_val, status

def get_frame_blur(gray_image):
    """
    Calculates the motion blur over the entire frame.
    Expects a grayscale image.
    Lower value = More Blur. Higher value = Crisper.
    """
    # Calculate Laplacian variance directly on the provided grayscale image
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    if blur_score < 60:
        status = "Blurry"
    else:
        status = "Clear"

    return blur_score, status