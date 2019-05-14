import numpy as np

'''
Description:
    An implementation of a traditional neural network with an input layer, output
        layer, and a single hidden layer.
'''
class NeuralNetwork:
    '''
    Description:
        Constructor for the neural network object.
    Parameters:
        in_nodes: number of input nodes / ie size of input
        out_nodes: number of output nodes. For classification, this will be the
                    number of potential output classes
        hidden: number of desried nodes in the hidden layer
    Returns:
        A neural network object
    '''
    def __init__(self, in_nodes=784, out_nodes=10, hidden=100):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.hidden= hidden
        self.initialize_params()

    '''
    Description:
        Initializes the weights for the neural network. To simplify, this model
            assumes a bias of 0, and therefore the bias is not included
    Parameters:
        in_nodes: number of input nodes / ie size of input
        out_nodes: number of output nodes. For classification, this will be the
                    number of potential output classes
        hidden: number of desried nodes in the hidden layer
    '''
    def initialize_params(self):
        # suppress numpy overflow warnings. Since the exponential function is
        #   is in the denomitor, this overflow to infinity will be driven to zero,
        #   as desried, and can therefore be ignored
        self.old_settings = np.seterr(over='ignore')

        # There are two sets of weights. The first is from the input to the hidden
        #   layer and the second from the hidden to the output layer
        self.w1 = np.random.randn(self.hidden, self.in_nodes)
        self.w2 = np.random.randn(self.out_nodes, self.hidden)


    '''
    Description:
        Does a single training pass through a given set.
    Parameters:
        data: the data set over which the training is to be conducted
        labels: the corresponding labels for each element of the data
            The data and label areas should be of the same length with each
                data index corresponding to the label of the same index
        params: if passed, the internal weights will be overwritten by passed weights
                    this is used for resuming from a previous training point
    Return:
        weights: returns an array that contains the two sets of weights that have
                    been trained to
    '''
    def train(self, data, labels, params=None):
        curr = 1    # used for printing progress
        if params is not None:  # check if params were passed
            [w1, w2] = params
            self.w1 = w1
            self.w2 = w2

        # iterate through the provided data and labels
        for img, label in zip(data, labels):
            # update status
            self.print_status("Training", len(data), curr)

            # conduct thye forward pass
            [layer1, output] = self.forward(img)

            # conduct the back propogation
            self.backward(layer1, output, img, label)
            curr += 1
        return [self.w1, self.w2]   # return the final weights

    '''
    Description:
        Itertates through a passed testing set of data and returns the perdiction
            accuracy.
    Parameters:
        data: the data set over which the testing is to be conducted
        labels: the corresponding labels for each element of the data
            The data and label areas should be of the same length with each
                data index corresponding to the label of the same index
        params: if passed, the internal weights will be overwritten by passed weights
                    this is used for testing from previously saved weights
    Return:
        weights: returns the persentage of accurate perdictions
    '''
    def test(self, data, labels, params=None):
        if params is not None:  # check if params were passed
            [w1, w2] = params
            self.w1 = w1
            self.w2 = w2

        # initialize accuracy variables
        total_predictions = len(data)
        correct_predictions = 0
        curr = 1    # used for printing progress

        # iterate through testing data and update accuracy as necessary
        for img, label in zip(data, labels):
            # print status
             self.print_status("Testing", total_predictions, curr)

             # conduct forward pass
             [layer1, output] = self.forward(img)

             # check if perdiction was correct
             ans = np.argmax(label)
             predict = np.argmax(output)
             if ans == predict:
                 correct_predictions += 1
             curr += 1
        # return accuracy
        return float(correct_predictions) / total_predictions

    '''
    Description:
        Given a single input and weights, perdicts the input class.
    Parameters:
        params: the array of weights
        img: the single data point to be predicted
    Return:
        class array: returns an array of probabilities representing the percentage
                        chance that the input belongs to each class
    '''
    def predict(self, params, img):
        [w1, w2] = params
        self.w1 = w1
        self.w2 = w2
        [layer1, output] = self.forward(img)
        return output

    '''
    Description:
        Prints the status of an operation.
    Parameters:
        type: string representing type of operation being conducted
        total: total number of inputs to operate over
        current: current input being operated on
    '''
    def print_status(self, type, total, current):
        percent_done = int(20 * float(current) / total)
        status = "[" + "|" * (percent_done) + " " * (20 - (percent_done)) + "]"
        print(type + " " + status + " {}%\t\t".format(percent_done * 5), end="\r")

    '''
    Description:
        Conducts the forward operation through the neural network for a single
            input.
    Parameters:
        img: input to process
    Return:
        returns an array containing the hidden layer and the output layer
    '''
    def forward(self, img):
        # each layer is created by taking the dot product of the previous layer
        #   and the weights associated with the transition. The sigmoid function
        #   is then applied to that to squeeze the values to between 0 and 1
        layer1 = self.sigmoid(np.dot(self.w1, img))
        output = self.sigmoid(np.dot(self.w2, layer1))
        return [layer1, output]

    '''
    Description:
        Conducts the backward operation through the neural network for a single
            input.
    Parameters:
        layer1: the hidden layer of the network
        output: the output layer of the network
        img: input being processed
        label: the correct class of the given img
    '''
    def backward(self, layer1, output, img, label):
        # get the difference of the output vs expected output
        diff = label - output

        # reverse the sigmoid of output by taking its derivative
        d_out = 2 * diff * self.sigmoid_deriv(output)

		# get partial deriviatives for the second set of weights
        d_w2 = np.dot(d_out, layer1.T)

        # pass the derivatives to the hidden layer
        d_l1 = np.dot(self.w2.T, d_out)
        # get the partial derivtatives of the first set of weights
        d_w1 = np.dot(d_l1 * self.sigmoid_deriv(layer1), img.T)

        # adjust the weights by their corresponding derivatives
        self.w1 += d_w1
        self.w2 += d_w2

    '''
    Description:
        Takes an array and applies the sigmoid function to it.
    Parameters:
        x: the array to apply the sigmoid to
    Return:
        a new array with the sigmoid applied
    '''
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    '''
    Description:
        Takes an array and applies the derivative of the sigmoid function to it.
    Parameters:
        x: the array to apply the derivative of the sigmoid to
    Return:
        a new array with the derivative of the sigmoid applied
    '''
    def sigmoid_deriv(self, x):
        return x * (1.0 - x)