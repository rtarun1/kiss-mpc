import cv2
import numpy as np
import sys

# --- CONFIGURATION ---
IMAGE_PATH = "rrc_lab.pgm" # <--- CHANGE THIS TO YOUR IMAGE FILE
MIN_RADIUS = 1 # Don't draw circles smaller than this radius
# ---------------------

# 1. Load your custom image
# cv2.IMREAD_GRAYSCALE loads the image in black and white
original_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if original_image is None:
    print(f"Error: Could not load image at {IMAGE_PATH}")
    sys.exit()

# 2. Pre-process the image: Ensure it's binary (0 and 255)
# We use a threshold to clean up the image. Pixels with a value > 127
# become 255 (white), and the rest become 0 (black).
_, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)

# This is our output image where we'll draw the circles
# Let's make the circles colored for better visualization
output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# 3. Invert the image
# cv2.distanceTransform finds the distance to the nearest 0 (black).
# We invert the image so our black areas become white for the function to process.
inverted_binary = cv2.bitwise_not(binary_image)

# 4. Calculate the Distance Transform
dist_transform = cv2.distanceTransform(inverted_binary, cv2.DIST_L2, 5)

# 5. Iteratively find the largest circle and draw it
while True:
    # Find the brightest point (max distance) in the transform map
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist_transform)

    # Break the loop if the max distance is too small to draw a circle
    if maxVal < MIN_RADIUS:
        break

    # The max value is the radius of the largest inscribed circle
    radius = int(maxVal)
    center = maxLoc # (x, y) coordinates

    # Draw the circle on our output image
    # Generate a random color for visualization
    color = np.random.randint(0, 255, size=3).tolist()
    cv2.circle(output_image, center, radius, color, -1) # -1 thickness fills the circle

    # Erase the area of the drawn circle from the distance transform map
    # This prevents drawing overlapping circles in the next iteration
    cv2.circle(dist_transform, center, radius, 0, -1)


# Display the results
cv2.imshow("Your Original Image", original_image)
cv2.imshow("Binary Version", binary_image)
cv2.imshow("Circles Filled", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the result
cv2.imwrite("circles_filled_output.png", output_image)
print("Result saved as circles_filled_output.png")