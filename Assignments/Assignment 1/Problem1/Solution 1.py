import cv2
from matplotlib import pyplot as plt
import numpy as np

# Define Image Path
carinaNebulaImg = cv2.imread("carinaNebula.png")

# Display Image
cv2.imshow("Original Image", carinaNebulaImg)
cv2.waitKey(0)  # Wait for user to exit image

grayScaleImage = cv2.cvtColor(carinaNebulaImg, cv2.COLOR_BGR2GRAY)  # Convert colored image to gray scale image

# Show gray level image and wait for exit by user
cv2.imshow("Gray Scale Image", grayScaleImage)
cv2.waitKey(0)

# Calculate Histogram
histogram = cv2.calcHist([grayScaleImage],[0],None, [256],[0,256]) # Calculate histogram with gray scale image and 256 levels of gray

# Plot and show histogram
plt.plot(histogram)
plt.show()

# Resize Image
resizedCarina = cv2.resize(carinaNebulaImg, (200, 200))   # Resize Image to 200x200

# Show resized image and wait for exit by user
cv2.imshow("Resized Image", resizedCarina)
cv2.waitKey(0)

equalizedHistogram = cv2.equalizeHist(grayScaleImage)   # Equalize Histogram
cv2.imshow("Equalized Histogram", equalizedHistogram)    # Show new image comparing two images
cv2.waitKey(0)

# Create two images with different thresholds
ret, thresh1 = cv2.threshold(grayScaleImage, 70, 255, 0)
ret, thresh2 = cv2.threshold(grayScaleImage, 100, 255, 0)

# Show Binary image and wait for user to exit
comparedThreshold = np.hstack((thresh1, thresh2))    # Put threshold of 70 and 100 next to each other to compare
# Show comparing image and wait for user to exit
cv2.imshow("Binary Image", comparedThreshold)
cv2.waitKey(0)

# Addition
venusImg = cv2.imread("venus.png")  # Read in new image (Venus Image)

resizedVenus = cv2.resize(venusImg, (200,200)) # Resize Venus image to allow for addition and subtraction

carinaNebulaPlusVenus = cv2.add(resizedVenus, resizedCarina)    # Add Venus image to Carina Nebula image

# Show image and wait for user to exit
cv2.imshow("Carina Nebula + Venus", carinaNebulaPlusVenus)
cv2.waitKey(0)

# Subtraction

carinaNebulaMinusVenus = cv2.subtract(resizedVenus, resizedCarina)  # Subtract Carina Nebula image from Venus image

# Show image and wait for user to exit
cv2.imshow("Carina Nebula - Venus", carinaNebulaMinusVenus)
cv2.waitKey(0)






