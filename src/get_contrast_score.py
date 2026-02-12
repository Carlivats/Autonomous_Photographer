import cv2
import numpy as np

def get_contrast_score(roi):
    """
    Calculates the RMS (Root Mean Square) contrast of an image.
    Higher value = Higher contrast.
    """
    # Calculate Standard Deviation (RMS Contrast)
    # std_dev measures how spread out the pixel brightness values are.
    _, contrast = cv2.meanStdDev(roi)

    # The standard deviation is our contrast metric.

    # We flatten the result to get a simple float number.
    contrast = contrast[0][0]

    if contrast < 30: status = "Very Low"
    elif contrast < 50: status = "Low/Med"
    elif contrast < 80: status = "Good"
    else: status = "High"
    
    return contrast, status

# --- Example ---
if __name__ == "__main__":
    frame = cv2.imread('./images/test.jpg') 
    
    contrast_score = get_contrast_score(frame)
    print(f"Contrast Score: {contrast_score:.2f}")