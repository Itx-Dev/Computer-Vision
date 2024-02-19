import cv2 as cv
import numpy as np
from scipy.fftpack import dct, idct

def performDCT(givenImage):
    # Create zero matrix with the same size as the image
    dctImage = np.zeros_like(givenImage)
    # Perform DCT in size 8 blocks across the entire image
    for i in range(0, rows, blockSize):
        for j in range(0, cols, blockSize):
            # Find 8x8 Block in the image
            block = givenImage[i:i + blockSize, j:j + blockSize]
            # Perform DCT on 8x8 block of the image
            dctBlock = dct(dct(block.T, norm='ortho').T, norm='ortho')
            # Set dctImage to values of dctBlock
            dctImage[i:i + blockSize, j:j + blockSize] = dctBlock

    return dctImage

def InverseDCT(compressedImage):
    # Perform IDCT in size 8 blocks across the entire image
    recoverImage = np.zeros_like(compressedImage)
    for i in range(0, rows, blockSize):
        for j in range(0, cols, blockSize):
            # Find 8x8 Block in the image
            block = compressedImage[i:i + blockSize, j:j + blockSize]
            inverseDCT = idct(idct(block.T, norm='ortho').T, norm='ortho')
            recoverImage[i:i + blockSize, j:j + blockSize] = inverseDCT

    return recoverImage

# Read image in as grayscale image
image = cv.imread('../pokemonImage.jpg', cv.IMREAD_GRAYSCALE)

# Turn image into float matrix
imageFloatMatrix = np.float32(image)

# Define block size and image dimensions
blockSize = 8
rows, cols = image.shape

# Perform Discrete Cosine Transform
DCT = performDCT(imageFloatMatrix)

# Perform logarithmic  processing
loggedImage = np.log1p(np.abs(DCT.astype(np.float32)))

# Perform Inverse Discrete Cosine Transform
recoveredImage = InverseDCT(DCT)

# Display the original and recovered images
cv.imshow('Recovered Image', recoveredImage.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()