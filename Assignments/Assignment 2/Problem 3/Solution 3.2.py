import cv2 as cv

# Read in image
img = cv.imread('ImageHW2B.jpg')

# Convert image to gray scale
grayScaleImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

equalizedHist = cv.equalizeHist(grayScaleImg)

# Show Image
cv.imshow("Revealed Message", equalizedHist)
cv.waitKey(0)