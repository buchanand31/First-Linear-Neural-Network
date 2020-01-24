import numpy as np

# Will train a neural network to do the mathematical equation: 2x + 3y
# This is a trivial example of a linear expression, this could be done with linear regression also, but I am
# simply testing out neural networks

class neural_network:
    def __init__(self):
        np.random.seed(1)
        # Consider a single perceptron, with 3 inputs and 1 output and assign a random weight
        self.weights = 2 * np.random.random((2, 1)) - 1
        
    def train(self, inputs, outputs, number):
        for i in range(number):
            output = self.think(inputs)
            error = outputs - output
            adjustment = 0.01 * np.dot(inputs.T, error)
            self.weights += adjustment
            
    def think(self, inputs):
        # Get the dot product of the inputs and the weights
        # A neural equation multiples the inputs by the weights and adds these up, hence the dot product
        return np.dot(inputs, self.weights)
    
neural_network = neural_network()

# The training set
inputs = np.array([[2, 3], [1, 1], [5, 2], [8, 3], [4, 2]])
outputs = np.array([[13, 5, 16, 25, 14]]).T

# Training the neural network using the trainng set, 1,000 times
neural_network.train(inputs, outputs, 1000)

# Asking the neural network for the output
x = int(input("First number:\n"))
y = int(input('Second number:\n'))

print(neural_network.think(np.array([x, y])))