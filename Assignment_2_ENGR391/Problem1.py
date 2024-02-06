import numpy as np
# Define file name for Image Matrix
inputMatrixFile = "Image Matrix.txt"
inputList = list()

# Define convolution mask
mask = np.array([-1, 0, 1])

# Open file to read-only
with open(inputMatrixFile, 'r') as f:
    # Iterate over each line and converting strings to ints
    for line in f:
        intList = [int(i) for i in line.split()]
        inputList.append(intList)   # Add list to another list for 2D array

matrix = np.array(inputList)
# Pad matrix with zeros
matrix = np.pad(matrix, 1, mode='constant', constant_values=0)
# Get rid of top and bottom rows being padded
matrix = np.array(matrix[1:-1])

def convolution(image, givenMask):
    answerArray = np.empty([0,5])
    givenMask = np.flip(givenMask)

    #print(np.convolve(image, givenMask, mode="same"))

    for i in range(2,4):
        answer = np.sum(np.matmul(image[:i], givenMask[-i:]))
        answerArray = np.append(answerArray, answer)

    for i in range(1,2):
        answer = np.sum(np.matmul(image[i:], givenMask[:-i]))
        answerArray = np.append(answerArray, answer)

    return answerArray

# Split Matrix into 4 separate ones
for i in range(0, 10):
    newMat = np.array_split(matrix[i], 4)
    # Limit Matrix to have min value of 0 and max value of 255
    print(newMat)
    print(newMat[0])


