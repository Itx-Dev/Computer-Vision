import numpy as np

## Example Mask and Image
mask = [-1, 0, 1]
matrix = [1, 1, 4,1,3,252,254,251,251,255,
         0,4,4,3,0,253,252,255,253,254,
         0,0,1,1,5,255,250,255,253,251,
         0,4,5,2,2,255,250,251,250,253,
         3,4,1,5,3,253,255,250,252,255,
         3,4,1,1,2,255,254,254,255,253,
         5,2,2,0,0,250,255,250,250,254,
         4,3,1,5,3,254,255,254,252,255,
         1,2,1,4,3,252,254,252,253,255,
         2,0,4,3,1,251,255,253,251,250]
## for 1x3 mask
def convolution(image, mask):
    # Define size of rows and cols
    rows, cols = image.shape
    size = mask.shape[0]
    # Set up array for storage of answers
    answer = np.zeros((rows, cols - size + 1))

    # Loop over to perform convolution operation
    for i in range(rows):
        for j in range(cols - size + 1):
            answer[i, j] = np.sum(image[i, j:j+size] * mask)

    # Return answer
    return answer
