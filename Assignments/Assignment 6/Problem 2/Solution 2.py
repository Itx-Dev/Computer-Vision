import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, inputs, learningRate=0.01, iteration=100, threshold=0.01):
        self.inputs = inputs
        self.learningRate = learningRate
        self.iterations = iteration
        self.weights = np.zeros(inputs)
        self.bias = 0
        self.threshold = threshold

    def predict(self, inputs):
        activation = np.dot(self.weights, inputs) + self.bias
        return 1 if activation >= 0 else 0

    def train(self, trainingData, labels):
        for iteration in range(self.iterations):
            weightChange = np.zeros_like(self.weights)
            biasChange = 0
            for inputs, label in zip(trainingData, labels):
                prediction = self.predict(inputs)
                weightChange += (label - prediction) * inputs
                biasChange += (label - prediction)
            self.weights += self.learningRate * weightChange
            self.bias += self.learningRate * biasChange

            # Check convergence criterion
            if np.linalg.norm(weightChange) + abs(biasChange) < self.threshold:
                print("Converged at iteration", iteration + 1)
                break

    def separation_line(self, x):
        return (-self.weights[0] * x - self.bias) / self.weights[1]


# Main loop
if __name__ == "__main__":
    # Define classes
    class1 = np.array([[5, 2], [6, 3], [5, 4], [6, 4]])
    class2 = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])

    # Combine classes and labels
    trainingData = np.vstack((class1, class2))
    desiredResults = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    # Initialize Perceptron
    perceptron = Perceptron(inputs=2, threshold=0.001)

    # Train Perceptron
    perceptron.train(trainingData, desiredResults)

    # Display trained weights and bias
    print("Trained Weights:", perceptron.weights)
    print("Trained Bias:", perceptron.bias)

    # Plot data points
    plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Class 1')
    plt.scatter(class2[:, 0], class2[:, 1], color='red', label='Class 2')

    # Plot separation line
    x_values = np.linspace(0, 5, 100)
    y_values = perceptron.separation_line(x_values)
    plt.plot(x_values, y_values, color='green', linestyle='--', label='Separation Line')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Classification with Separation Line')
    plt.legend()
    plt.grid(True)
    plt.show()