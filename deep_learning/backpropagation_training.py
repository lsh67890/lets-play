import time
import numpy as np

# utils
def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)

# Sigmoid
class Sigmoid:
    def __init__(self):
        self.last_o = 1 # initialise to 1

    def __call__(self, x):
        self.last_o = 1 / (1.0 + np.exp(-x))
        return self.last_o

    def grad(self): # sigmoid(x)(1 - sigmoid(x)) saved to use as dynamic programming
        return self.last_o * (1 - self.last_o)

# mean squared error
class MeanSquaredError:
    def __init__(self):
        # gradient
        self.dh = 1 # save this to use in chain rule
        self.last_diff = 1

    def __call__(self, h, y): # 1/2 * mean ((h - y)^2)
        self.last_diff = h - y
        return 1 / 2 * np.mean(np.square(h-y))

    def grad(self): # h - y
        return self.last_diff

# neuron
class Neuron:
    def __init__(self, W, b, a_obj):
        #model params
        self.W = W
        self.b = b
        self.a = a_obj()

        # gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dh = np.zeros_like(_t(self.W))

        # save the last x to use in differentiation
        self.last_x = np.zeros((self.W.shape[0]))
        self.last_h = np.zeros((self.W.shape[1]))

    def __call__(self, x):
        self.last_x = x
        self.last_h = _m(_t(self.W), x) + self.b
        return self.a(self.last_h)

    def grad(self): # dy/dh = W
        # differentiate with the previous differentiation (h)
        return self.W * self.a.grad()

    def grad_W(self, dh): # gradient differentiated by W
        grad = np.ones_like(self.W)
        grad_a = self.a.grad()
        for j in range(grad.shape[1]): # dy/dw = x
            grad[:, j] = dh[j] * grad_a[j] * self.last_x
        return grad

    def grad_b(self, dh): # dy / dh = 1
        return dh * self.a.grad()

# DNN
class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activation=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # first hidden layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # hidden layers
        for index in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # output layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, loss_obj):
        loss_obj.dh = loss_obj.grad()
        self.sequence.append(loss_obj)
        # back propagation loop
        for i in range(len(self.sequence) - 1, 0, -1): # looping backwards
            l1 = self.sequence[i]
            l0 = self.sequence[i - 1]

            l0.dh = _m(l0.grad(), l1.dh)
            l0.dW = l0.grad_W(l1.dh)
            l0.db = l0.grad_b(l1.dh)

        self.sequence.remove(loss_obj)

def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y) # forward inference
    network.calc_gradient(loss_obj) # backpropagation
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss

# test
x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))
t = time.time()
# try with larger hidden_depth and num_neuron
dnn = DNN(hidden_depth=5, num_neuron=32, input=10, output=2, activation=Sigmoid)
loss_obj = MeanSquaredError()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)
    print('Epoch {}: test loss {}'.format(epoch, loss))
print('{} seconds elapsed'.format(time.time() - t))