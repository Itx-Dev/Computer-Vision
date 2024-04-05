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
    classA = np.array([[0.2801, 0.05977], [0.3254, 0.5610], [0.2829, 0.5972], [0.2898, 0.5754], [0.6665, 0.2978], [0.4495, 0.4375], [0.4613, 0.4485]])
    classB = np.array([[0.4252, 0.5667], [0.4183, 0.6134], [0.2681, 0.8249], [0.4700, 0.5498], [1.2044, 0.3161], [0.9762, 0.4029], [0.7326, 0.4645]])

    # Combine classes and labels
    trainingData = np.vstack((classA, classB))
    desiredResults = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    # Initialize Perceptron
    perceptron = Perceptron(inputs=2, threshold=0.001)

    # Train Perceptron
    perceptron.train(trainingData, desiredResults)

    # Display trained weights and bias
    print("Trained Weights:", perceptron.weights)
    print("Trained Bias:", perceptron.bias)

    # Plot data points
    plt.scatter(classA[:, 0], classA[:, 1], color='blue', label='Class A')
    plt.scatter(classB[:, 0], classB[:, 1], color='red', label='Class B')

    # Plot separation line
    x_values = np.linspace(0, 1, 10)
    y_values = perceptron.separation_line(x_values)
    plt.plot(x_values, y_values, color='green', linestyle='--', label='Separation Line')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Classification with Separation Line')
    plt.legend()
    plt.grid(True)
    plt.show()