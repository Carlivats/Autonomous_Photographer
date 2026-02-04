import cv2
import numpy as np

def get_exposure_score(image):
    """
    Analyzes the brightness levels of the image.
    Returns the average brightness and an exposure status.
    """
    # 1. Convert to grayscale to get brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Calculate the Mean Brightness
    # 'mean_brightness' ranges from 0 (pure black) to 255 (pure white).
    mean_brightness, _ = cv2.meanStdDev(gray)

    # We flatten the result to get a simple float number.
    exposure_score = mean_brightness[0][0]

    # These thresholds can be adjusted based on future outdoor/indoor testing.
    if exposure_score < 50:
        status = "Underexposed (Too Dark)"
    elif exposure_score > 160:
        status = "Overexposed (Too Bright / Glare)"
    else:
        status = "Well-Exposed"

    return exposure_score, status

# --- Example ---
if __name__ == "__main__":
    frame = cv2.imread('./images/test.jpg') 

    brightness, exposure_label = get_exposure_score(frame)
    print(f"Average Brightness: {brightness:.2f}")
    print(f"Status: {exposure_label}")