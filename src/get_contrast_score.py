import cv2
import numpy as np

def get_contrast_score(image):
    """
    Calculates the RMS (Root Mean Square) contrast of an image.
    Higher value = Higher contrast.
    """
    # Convert to grayscale because contrast is about light intensity, not color.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Calculate Standard Deviation (RMS Contrast)
    # std_dev measures how spread out the pixel brightness values are.
    _, std_dev = cv2.meanStdDev(gray)

    # The standard deviation is our contrast metric.
    # We flatten the result to get a simple float number.
    contrast_score = std_dev[0][0]
    
    return contrast_score

# --- Example ---
if __name__ == "__main__":
    frame = cv2.imread('./images/test.jpg') 
    
    contrast_score = get_contrast_score(frame)
    print(f"Contrast Score: {contrast_score:.2f}")