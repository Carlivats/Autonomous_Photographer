import cv2
import numpy as np

def get_subject_sharpness(image, x, y, w, h):
    """
    Calculates sharpness specifically for the detected subject.
    """
    # 1. Boundary Protection
    # Ensure coordinates are within the image frame
    img_h, img_w = image.shape[:2]
    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(img_w, x + w), min(img_h, y + h)

    # 2. Crop the image to the bounding box
    subject_roi = image[y_start:y_end, x_start:x_end]
    if subject_roi.size == 0:
        return 0.0

    # 3. Convert crop to grayscale
    gray_subject = cv2.cvtColor(subject_roi, cv2.COLOR_BGR2GRAY)

    # 4. Calculate Sharpness Score (Laplacian Variance)
    sharpness_score = cv2.Laplacian(gray_subject, cv2.CV_64F).var()

    return sharpness_score

# --- Example ---
if __name__ == "__main__":
    frame = cv2.imread('./images/test1.jpg') 
    # Example bounding box (x, y, w, h)
    x, y, w, h = 000, 50, 500, 500  
    
    sharpness_score = get_subject_sharpness(frame, x, y, w, h)
    print(f"Subject Sharpness Score: {sharpness_score:.2f}")

    # Display the cropped subject for visual reference
    subject_roi = frame[y:y+h, x:x+w]
    cv2.imshow("Subject ROI", subject_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()