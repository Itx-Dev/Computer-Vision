import cv2
import numpy as np
import cv2 as cv

# Read Image
img = cv.imread('../edgeDetectionImage.PNG')

# Convert Image to Gray Scale
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# X Component of Sobel Mask (X Component finds vertical lines)
xSobelMask = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

# Y Component of Sobel Mask (Y Component find horizontal lines)
ySobelMask = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

# Apply X Component of Sobel Mask to gray image
sobelX = cv2.filter2D(grayImage, cv2.CV_64F, xSobelMask)
# Show X Component of Sobel Mask
cv.imshow("Sobel X", sobelX)
cv.waitKey(0)

# Apply Y Component of Sobel Mask to gray image
sobelY = cv2.filter2D(grayImage, cv2.CV_64F, ySobelMask)
# Show Y Component of Sobel Mask
cv.imshow("Sobel Y", sobelY)
cv.waitKey(0)

# Find the magnitude of the sobel mask
sobelMag = np.sqrt(sobelX**2 + sobelY**2)
# Show Complete Sobel Mask
cv.imshow("Sobel Magnitude", sobelMag)
cv.waitKey(0)


# Define X Component of Prewitt Mask
xPrewittMask = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])

# Apply X Component of Prewitt Mask
prewittX = cv2.filter2D(grayImage, cv2.CV_64F, xPrewittMask)
# Display Image
cv.imshow("Prewitt X", prewittX)
cv.waitKey(0)

# Define Y Component of Prewitt Mask
yPrewittMask = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])

# Apply Y Component of Prewitt Mask
prewittY = cv2.filter2D(grayImage, cv2.CV_64F, yPrewittMask)
# Display Image
cv.imshow("Prewitt Y", prewittY)
cv.waitKey(0)

# Find Magnitude of Prewitt Mask
prewittMag = np.sqrt(prewittX**2 + prewittY**2)
# Display Final Result
cv.imshow("Prewitt Magnitude", prewittMag)
cv.waitKey(0)

# Compare both mask
combinedMasks = np.concatenate((sobelMag, prewittMag), axis=1)

cv.imshow("Sobel vs Prewitt Magnitude", combinedMasks)
cv.waitKey(0)

