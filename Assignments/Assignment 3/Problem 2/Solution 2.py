import numpy as np
import cv2 as cv
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt

# Import image
img = cv.imread('../edgeDetectionImage.PNG')

grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# X Component of Sobel Mask (X Component finds vertical lines)
xSobelMask = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

# Y Component of Sobel Mask (Y Component find horizontal lines)
ySobelMask = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

# Apply 2D fft to x and y components of sobel mask
# S= parameter allows for zero padding by passing in the shape of the original image
xSobelFFT = fft2(xSobelMask, s=grayImage.shape)
ySobelFFT = fft2(ySobelMask, s=grayImage.shape)
# Apply 2D fft to gray Image
imageFFT = fft2(grayImage)

# Apply filter to image in frequency domain
xResult = imageFFT * xSobelFFT
yResult = imageFFT * ySobelFFT
# Convert back to space domain for displaying
xResult = ifft2(xResult).real
yResult = ifft2(yResult).real
# Find magnitude of sobel mask
sobelMaskMagnitude = np.sqrt(xResult**2 + yResult**2)

# Show X Result
cv.imshow('X Sobel Mask; Frequency', xResult)
cv.waitKey(0)
# Show Y Result
cv.imshow('Y Sobel Mask; Frequency', yResult)
cv.waitKey(0)
# Show magnitude
cv.imshow("Magnitude of Sobel Mask; Frequency", sobelMaskMagnitude)
cv.waitKey(0)