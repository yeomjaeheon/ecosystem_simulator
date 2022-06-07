import numpy as np

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

class nn:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_size = input_size * 2 + hidden_size + output_size
        self.neurons = np.zeros(input_size * 2 + hidden_size + output_size)
        self.mask = np.zeros((self.total_size, input_size + hidden_size + output_size))
        for i in range(0, input_size):
            self.mask[i][i] = 1
        for i in range(0, input_size + hidden_size + output_size):
            for j in range(0, input_size + hidden_size + output_size):
                if j != i:
                    self.mask[input_size + j][i] = 1
        self.weight = np.random.normal(0, 10 / (input_size + hidden_size + output_size), (input_size * 2 + hidden_size + output_size, input_size + hidden_size + output_size)) * self.mask

    def calcul(self, input_data):
        for i in range(0, self.input_size):
            self.neurons[i] = input_data[i]
        state_next = sigmoid(np.dot(self.neurons, self.weight))
        for i in range(0, self.input_size + self.hidden_size + self.output_size):
            self.neurons[self.input_size + i] = state_next[i]
        return self.neurons[self.input_size * 2 + self.hidden_size : self.total_size]

    def mutate(self, c):
        self.weight += np.random.normal(0, 1 / (self.input_size + self.hidden_size + self.output_size), (self.total_size, self.input_size + self.hidden_size + self.output_size)) * self.mask

                
