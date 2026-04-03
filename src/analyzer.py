# analyzer.py
import cv2

from algorithms.get_exposure_score import get_exposure_score

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