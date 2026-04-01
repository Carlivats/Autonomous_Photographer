import cv2
import numpy as np

def get_exposure_score(roi):
    """
    Analyzes the brightness levels of the image.
    Returns the average brightness and an exposure status.
    """
    # 1. Calculate the Mean Brightness
    # 'mean_brightness' ranges from 0 (pure black) to 255 (pure white).
    mean_brightness, _ = cv2.meanStdDev(roi)

    # We flatten the result to get a simple float number.
    exposure = mean_brightness[0][0]

    # These thresholds can be adjusted based on future outdoor/indoor testing.
    if exposure < 50:
        status = "Underexposed"
    elif exposure > 160:
        status = "Overexposed"
    else:
        status = "Good"

    return exposure, status

# --- Example ---
if __name__ == "__main__":
    frame = cv2.imread('./images/test.jpg') 

    exposure_score, exposure_label = get_exposure_score(frame)
    print(f"Average Exposure: {exposure_score:.2f}")
    print(f"Status: {exposure_label}")