import time
import numpy as np

epsilon = 0.0001

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(h, y):
    return 1 / 2 * np.mean(np.square(h - y))

class Neuron:
    def __init__(self, W, b, a):
        self.W = W
        self.b = b
        self.a = a

        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) # bias

    def __call__(self, x):
        # activation((W^T)x + b)
        return self.a(np.matmul(np.transpose(self.W), x) + self.b)

class DNN:
    def __init__(self, hidden_depth, num_neuron, num_input, num_output, activation=sigmoid):
        def init_var(i, o): # initialise w and b
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # first hidden layer
        W, b = init_var(num_input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # hidden layers
        for _ in range(hidden_depth - 1):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # output layer
        W, b = init_var(num_neuron, num_output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calculate_gradient(self, x, y, loss_func):
        def get_new_sequence(layer_idx, new_neuron):
            new_sequence = list()
            for i, layer in enumerate(self.sequence):
                if i == layer_idx:
                    new_sequence.append(new_neuron)
                else:
                    new_sequence.append(layer)
            return new_sequence

        def evaluate_sequence(x, sequence):
            for layer in sequence:
                x = layer(x)
            return x

        # self(x) is what we assume
        loss = loss_func(self(x), y)

        # going through all weights and biases
        for layer_id, layer in enumerate(self.sequence): # iterate layer
            for w_i, w in enumerate(layer.W): # iterate weight row
                for w_j, ww in enumerate(w): # iterate weight col
                    W = np.copy(layer.W)
                    W[w_i][w_j] = ww + epsilon

                    new_sequence = get_new_sequence(layer_id, Neuron(W, layer.b, layer.a))
                    h = evaluate_sequence(x, new_sequence)
                    numerical_gradient = (loss_func(h, y) - loss) / epsilon # (f(x+eps) - f(x)) / epsilon
                    layer.dW[w_i][w_j] = numerical_gradient

                for b_i, bb in enumerate(layer.b):
                    b = np.copy(layer.b)
                    b[b_i] = bb + epsilon

                    new_sequence = get_new_sequence(layer_id, Neuron(layer.W, layer.b, layer.a))
                    h = evaluate_sequence(x, new_sequence)

                    numerical_gradient = (loss_func(h, y) - loss) / epsilon
                    layer.db[b_i] = numerical_gradient

        return loss

def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = network.calculate_gradient(x, y, loss_obj)
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss

# test
x = np.random.normal(0.0, 1.0, (10, ))
y = np.random.normal(0.0, 1.0, (2, ))

dnn = DNN(hidden_depth=5, num_neuron=32, num_input=10, num_output=2, activation=sigmoid)

t = time.time()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, mean_squared_error, 0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds taken'.format(time.time() - t))
