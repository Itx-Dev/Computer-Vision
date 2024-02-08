import numpy as np
import cv2 as cv

# Define Image Path and Window Name
imageFile = "pillarsOfCreation.jpg"
windowName = "Pillars of Creation"

# Read Image In
image = cv.imread(imageFile)

# Convert Image to gray scale
grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#cv.imshow(windowName, grayImage)  # Show Image
#cv.waitKey(0)  # Wait for exit


####
def saltAndPepper(image, strength):
    # Create random array with size of image, where values are in range of [0,1]
    randomArrayValue = np.random.rand(image.shape[0], image.shape[1])
    # Make copy of image
    noisyImage = image.copy()
    # If randomArrayValue < strength set that pixel to 0
    noisyImage[randomArrayValue < strength] = 0
    # If randomArrayValue > 1 - strength set that pixel to 1
    noisyImage[randomArrayValue > 1 - strength] = 1

    return noisyImage


# Call Salt and Pepper funtion
saltAndPepperImage = saltAndPepper(grayImage, 0.1)


# Display Image
#cv.imshow("Salt and Pepper Image", saltAndPepperImage)
#cv.waitKey(0)

###
def medianBlur(image):
    # Create a 3x3 mask
    mask = np.ones((3,3))
    # Define height and width of image
    height, width = image.shape
    # Loop over entire image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract 3x3 neighborhood of image
            neighborhoodMatrix = image[i - 1: i + 2, j - 1: j + 2]
            # Find median value of 3x3 neighborhood
            medianValue = np.median(neighborhoodMatrix)
            # Set center value equal to media value
            neighborhoodMatrix[1][1] = medianValue
    # Return image
    return image

# Run median blur function
fixedImage = medianBlur(saltAndPepperImage)

# Apply median filter to salt and pepper image
'''fixedImage = cv.medianBlur(saltAndPepperImage, 5)
'''
# Display Image
cv.imshow('Fixed Image', fixedImage)
cv.waitKey(0)

