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
