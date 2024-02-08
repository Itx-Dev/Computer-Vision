import numpy as np

def convolution(image, mask):
    # Define size of rows and cols
    rows, cols = matrix.shape
    size = mask.shape[0]
    # Set up array for storage of answers
    answer = np.zeros((rows, cols - size + 1))

    # Loop over to perform convolution operation
    for i in range(rows):
        for j in range(cols - size + 1):
            answer[i, j] = np.sum(matrix[i, j:j+size] * mask)

    # Return answer
    return answer

# Define file name for Image Matrix
inputMatrixFile = "Image Matrix.txt"
# Create empty list to store .txt file content
inputList = list()

# Define convolution mask
mask = np.array([-1, 0, 1])

# Open file to read-only
with open(inputMatrixFile, 'r') as f:
    # Iterate over each line and converting strings to ints
    for line in f:
        intList = [int(i) for i in line.split()]
        inputList.append(intList)   # Add list to another list for 2D array

# Create a numpy array from string list, with data type integer
matrix = np.array(inputList, dtype=int)
# Pad matrix with zeros
matrix = np.pad(matrix, 1, mode='constant', constant_values=0)
# Get rid of top and bottom rows being padded
matrix = np.array(matrix[1:-1])

# Call convolution function with original image and mask
answer = convolution(matrix, mask)
# Clip the convolution matrix so all negative values are 0 and values over 255 are 255
clippedAnswer = np.clip(answer, 0, 255)

print(clippedAnswer)


