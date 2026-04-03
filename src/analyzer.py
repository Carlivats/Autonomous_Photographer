# analyzer.py
import cv2

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

def get_subject_exposure(image, x1, y1, x2, y2):
    pass