import cv2 as cv

# Read in image
img = cv.imread('ImageHW2B.jpg')

# Convert image to gray scale
grayScaleImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Threshhold Image
_, threshImg = cv.threshold(grayScaleImg, 201, 220, cv.THRESH_BINARY_INV)

# Blur image that is thresholded
blurredThreshImg = cv.medianBlur(threshImg, 5)

# Show Image
cv.imshow("Revealed Message", blurredThreshImg)
cv.waitKey(0)