import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in image
fig1 = cv2.imread("../HW5Fig1.JPG")

# Define structuring element
structuringElement = np.ones((8,8), np.uint8)
# Erode the image
errodedImage = cv2.erode(fig1, structuringElement)
# Define subplot to display images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(fig1)
axes[0].set_title("Original Image")

axes[1].imshow(errodedImage)
axes[1].set_title("Erroded Image")

plt.show()

# Dilate Image
dilatedImage = cv2.dilate(fig1, structuringElement)

# Create a new figure and axes for the second set of images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(fig1)
axes[0].set_title("Original Image")

axes[1].imshow(dilatedImage)
axes[1].set_title("Dilated Image")

plt.show()

# Apply morphlogical gradient to image
morphedImage = cv2.morphologyEx(fig1, cv2.MORPH_GRADIENT, structuringElement)

# Create a new figure and axes for the second set of images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(fig1)
axes[0].set_title("Original Image")

axes[1].imshow(morphedImage)
axes[1].set_title("Morphed Image")

plt.show()

# Read in new image
fig2 = cv2.imread("../HW5Fig2.JPG")
# Create structuing elements for dilation and erosion operation
dilationStructuringElement = np.ones((10,10), np.uint8)
erosionStructuringElement = np.ones((12,12), np.uint8)
# Apply dilation to remove noise inside of B
fixedWithDilation = cv2.dilate(fig2, dilationStructuringElement)
# Apply erosion to remove the noise in the background and to make B the original size
fixedWithErosion = cv2.erode(fixedWithDilation, erosionStructuringElement)

# Create a new figure and axes for the new images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(fig2)
axes[0].set_title("Original Image")

axes[1].imshow(fixedWithErosion)
axes[1].set_title("Figure 2 Fixed")

plt.show()

# Read in new image
fig3 = cv2.imread("../HW5Fig3.JPG")

# Create structuing elements for dilation and erosion operation
dilationSE = np.ones((12,12), np.uint8)
erosionSE = np.ones((9,9), np.uint8)

# Apply erosion to remove the noise in the background and to make B the original size
erosionApplied = cv2.erode(fig3, erosionSE)
# Apply dilation to remove noise inside of B
dilationApplied = cv2.dilate(erosionApplied, dilationSE)


# Create a new figure and axes for the new images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(fig3)
axes[0].set_title("Original Image")

axes[1].imshow(dilationApplied)
axes[1].set_title("Figure 3 Fixed")

plt.show()








