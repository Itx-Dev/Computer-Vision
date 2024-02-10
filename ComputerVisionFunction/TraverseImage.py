import numpy as np

# Travel across image with 3x3 kernal
def traverseImage(image):
    # Create a 3x3 mask
    mask = np.ones((3, 3))

    # Define height and width of image
    height, width = image.shape
    # Loop over entire image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract 3x3 neighborhood of image
            neighborhoodMatrix = image[i - 1: i + 2, j - 1: j + 2]
