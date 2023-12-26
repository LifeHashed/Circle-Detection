import cv2
import numpy as np

# Load an image from file
image = cv2.imread('dustin-humes-oTwMc6H-1RE-unsplash.jpg')  # Replace 'your_image_path.jpg' with the path to your image

# resizing image
original_height, original_width = image.shape[:2]
new_width = 500
new_height = int((new_width / original_width) * original_height)

image = cv2.resize(image, (new_width, new_height))
# Check if image has been loaded
if image is None:
    print("Error: Could not read the image.")
else:
    print("Image has been successfully loaded.")
    # Further processing or display can be done here
    cv2.imshow('Loaded Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and help with circle detection
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Use HoughCircles to detect circles in the image
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=1,
    param1=30,
    param2=30,
    minRadius=1,
    maxRadius=100
)



# If circles are found, draw them on the original image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Fit an ellipse to each contour
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (255, 0, 0), 2)

# Display the result
cv2.imshow('Circle and ellipse Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

