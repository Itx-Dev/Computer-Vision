import numpy as np
import cv2 as cv

# Create a blank image with dtype np.uint8
D = np.zeros((200, 200), dtype=np.uint8)

# Draw lines of original image
D[50, :] = 255
D[150, :] = 255
D[:, 50] = 255
D[:, 150] = 255

# Copy original image and convert to be able to display color
colorImage = np.copy(D)
colorImage = cv.cvtColor(colorImage, cv.COLOR_GRAY2BGR)

# Perform edge detection
edges = cv.Canny(D, 50, 200, None, 3)

# Perform Hough Line Transform
lines = cv.HoughLines(edges, 1, np.pi / 180, 150)

# Iterate over each line from Hough's Transform and draw lines in color
for rAndTheta in lines:
    floatArray = np.array(rAndTheta[0], dtype=np.float64)
    r, theta = floatArray

    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * r
    y0 = b * r

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))

    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Draw lines from calculated points in color red
    cv.line(colorImage, (x1, y1), (x2, y2), (0, 0, 255), 2)


# Display the result
cv.imshow('Detected Lines', colorImage)
cv.waitKey(0)
cv.destroyAllWindows()