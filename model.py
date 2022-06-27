import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(model, X, Y):
    '''
    Plot the decision boundary of dataset for Logistic Regression
    :param model: LR or other model
    :param X: feature values of examples
    :param Y: the class of corresponding examples
    '''

    # set the min and max values and give it some padding
    x_min, x_max = X[0,:].min()-1, X[0,:].max()+1
    y_min, y_max = X[1,:].min()-1, X[1,:].max()+1
    h = 0.01
    # generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
    # predict the function value for the whole grid
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # plot the contour and training examples
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.title('Logtistic Regression')
    plt.scatter(X[0,:],X[1,:],c=Y.ravel(),cmap=plt.cm.Spectral)


def load_planar_dataset():
    '''
     Create the dataset we are going to use. This dataset contains two classes.
    :return: the data matrix of examples and its labels
    '''
    np.random.seed(1)
    m = 400             # number of examples
    N = int(m/2)        # number of points per class
    D = 2               # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4   # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2   # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def layer_size(X, Y):
    '''
    Build two layers' neural network.
    Input layer: only two variables x1 and x2
    Hidden Layer: 4 nodes
    Output Layer: only one output value y
    :param X: feature values of examples
    :param Y: the class of corresponding examples
    :return: the size of each layer
    '''

    n_x = X.shape[0]     # the size of input layer
    n_h = 4              # the size of hidden layer
    n_y = Y.shape[0]     # the size of output layer

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    '''
    initialize the model's parameters.
    :param n_x: the size of input layer
    :param n_h: the size of hidden layer
    :param n_y: the size of output layer
    :return: parameters
    '''

    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        'W1': w1,
        'W2': w2,
        'b1': b1,
        'b2': b2
    }

    return parameters


def sigmoid(x):
    '''
    Compute the sigmoid of x
    :param x: A scalar or numpy array of any size.
    :return: s -- sigmoid(x)
    '''
    s = 1/(1+np.exp(-x))
    return s


def forward_propagation(X, parameters):
    '''
    Forward propagation
    :param X: input data of size (n_x, m)
    :param parameters: python dictionary containing your parameters (output of initialization function)
    :return: The sigmoid output of the second activation; A dictionary containing "Z1", "A1", "Z2" and "A2"
    '''
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']     # output size: (n_h, 1)
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']    # output size: (n_y, 1)
    A2 = sigmoid(Z2)

    cache = {
        'Z1': Z1,
        'Z2': Z2,
        'A1': A1,
        'A2': A2
    }

    return A2, cache


def compute_cost(A2, Y, parameters):
    '''
    Implement this function to compute the value of the cost J.
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    :return: cross-entropy cost given equation
    '''

    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = - np.sum(logprobs) / m

    return cost


def backward_propagation(parameters, cache, X, Y):
    '''
    Implement the backward propagation using the instructions above.
    :param parameters: python dictionary containing our parameters
    :param cache: a dictionary containing "Z1", "A1", "Z2" and "A2".
    :param X: input data of shape (2, number of examples)
    :param Y: "true" labels vector of shape (1, number of examples)
    :return: python dictionary containing your gradients with respect to different parameters
    '''

    m = Y.shape[1]      # number of examples

    A2 = cache['A2']
    A1 = cache['A1']
    Z2 = cache['Z2']
    Z1 = cache['Z1']
    W2 = parameters['W2']       # size -> (n_y, n_h)
    W1 = parameters['W1']

    dZ2 = A2 - Y        # size -> (1, n_y)
    dW2 = np.multiply(np.dot(dZ2, A1.T), 1/m)       # size -> (n_y, n_h)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))         # size -> (n_h, 1)
    dW1 = np.multiply(np.dot(dZ1, X.T), 1/m)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    '''
    Updates parameters using the gradient descent update rule given above
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients
    :param learning_rate: learning step
    :return: python dictionary containing your updated parameters
    '''

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # update the value of parameters using gradent decent
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2
    }

    return parameters


def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
    '''

    :param X:
    :param Y:
    :param n_h:
    :param num_iterations:
    :param print_cost:
    :return:
    '''

    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.

    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions









