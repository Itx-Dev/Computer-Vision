import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in as grayscale
image = cv2.imread('../pokemonImage.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the image matrix to float 32
imageFloatMatrix = np.float32(image)

# Perform Discrete Cosine Transform
dctImage = cv2.dct(imageFloatMatrix)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(dctImage) + 1), cmap='jet')
plt.title('DCT')

plt.show()

# Keep (n1/2, n2/2) elements from the top left of ImageDCT
ImageDCTToPLeft1 = dctImage[:dctImage.shape[0] // 2, :dctImage.shape[1] // 2]

# Keep (n1/4, n2/4) elements from the top left of ImageDCT
ImageDCTToPLeft2 = dctImage[:dctImage.shape[0] // 4, :dctImage.shape[1] // 4]


# Perform Inverse DCT on ImageDCTToPLeft1 and ImageDCTToPLeft2
recoveredImage1 = cv2.idct(ImageDCTToPLeft1)
recoveredImage2 = cv2.idct(ImageDCTToPLeft2)

plt.subplot(2, 2, 1)
plt.imshow(np.log(np.abs(ImageDCTToPLeft1) + 1), cmap='jet')
plt.title('DCT of ImageDCTToPLeft1')

plt.subplot(2, 2, 3)
plt.imshow(recoveredImage1, cmap='gray')
plt.title('Inverse of ImageDCTToPLeft1')

plt.subplot(2, 2, 2)
plt.imshow(np.log(np.abs(ImageDCTToPLeft2) + 1), cmap='jet')
plt.title('DCT of ImageDCTToPLeft2')

plt.subplot(2, 2, 4)
plt.imshow(recoveredImage2, cmap='gray')
plt.title('Inverse of ImageDCTToPLeft2')

plt.show()

# Keep (n1/2, n2/2) elements from the bottom right of ImageDCT
ImageDCTBottomRight1 = dctImage[dctImage.shape[0] // 2:, dctImage.shape[1] // 2:]

# Keep (n1/4, n2/4) elements from the bottom right of ImageDCT
ImageDCTBottomRight2 = dctImage[dctImage.shape[0] // 4:, dctImage.shape[1] // 4:]

# Perform Inverse DCT on ImageDCTBottomRight1 and ImageDCTBottomRight2
recoveredImage1Bottom = cv2.idct(ImageDCTBottomRight1)
recoveredImage2Bottom = cv2.idct(ImageDCTBottomRight2)

plt.subplot(2, 1, 1)
plt.imshow(recoveredImage1Bottom, cmap='gray')
plt.title('Inverse of ImageDCTBottomRight1')

plt.subplot(2, 1, 2)
plt.imshow(recoveredImage2Bottom, cmap='gray')
plt.title('Inverse of ImageDCTBottomRight2')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()

plt.subplot(2, 1, 1)
plt.imshow(np.log(np.abs(ImageDCTBottomRight1) + 1), cmap='jet')
plt.title('DCT of Bottom Right 1')

plt.subplot(2, 1, 2)
plt.imshow(np.log(np.abs(ImageDCTBottomRight2) + 1), cmap='jet')
plt.title('DCT of Bottom Right 2')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()
