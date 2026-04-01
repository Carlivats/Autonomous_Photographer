import cv2

def get_blur_score(image):
    """
    Calculates the blurriness of an image using the Laplacian operator.
    Lower value = More Blur.
    Higher value = Sharper/Crisper.
    """

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Apply the Laplacian operator
    motion_blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # These thresholds can be adjusted based on future outdoor/indoor testing.
    if motion_blur_score < 60:
        status = "Blurry"
    else:
        status = "Clear"

    return motion_blur_score, status

# --- Example ---
if __name__ == "__main__":
    frame = cv2.imread('./images/test.jpg')
    
    motion_blur_score, status = get_blur_score(frame)
    print(f"Blur Score: {motion_blur_score:.2f}")
    print(f"Status: {status}")