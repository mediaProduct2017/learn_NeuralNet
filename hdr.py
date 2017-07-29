
# coding: utf-8

# In[ ]:

# hand-written digit recognition (hdr)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import optimize

import math
from collections import namedtuple
import math
import random
import os
import json

"""
This class does some initial training of a neural network for predicting drawn
digits based on a data set in data_matrix and data_labels. It can then be used to
train the network further by calling train() with any array of data or to predict
what a drawn digit is by calling predict().

The weights that define the neural network can be saved to a file, NN_FILE_PATH,
to be reloaded upon initilization.
"""

class HdrNeuralNetwork:
    WIDTH_IN_PIXELS = 20
    LEARNING_RATE = 0.1
    # for online learning
    NN_FILE_PATH = 'nn.json'

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, use_file=True):
        self.num_hidden_nodes = num_hidden_nodes
        self._use_file = use_file 
        # self.theta1
        # self.theta2
        
        self.LAMBDA = 0
        # for regularization of the cost function        
        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        self.sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)
        self.data_matrix = data_matrix # 2-D list
        self.data_labels = data_labels # 1-D list
        self.sample_num = len(self.data_labels)

        if (not os.path.isfile(HdrNeuralNetwork.NN_FILE_PATH) or not use_file):
            # it could also be self.NN_FILE_PATH
            # Step 1: Initialize weights to small numbers
            self.theta1 = self._rand_initialize_weights(self.num_hidden_nodes, 400+1)
            # num_hidden_nodes*401 matrix, the one is the bias
            self.theta2 = self._rand_initialize_weights(10, self.num_hidden_nodes+1)
            # 10*(num_hidden_nodes+1) matrix, the one is the bias

            # Train using sample data
            TrainData = namedtuple('TrainData', ['fig', 'label'])
            the_temp = tuple(np.row_stack((self.theta1.flatten(1).T, self.theta2.flatten(1).T)).T.tolist()[0])
            theta0 = np.asarray(the_temp) 
            args = tuple([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in range(self.sample_num)])
            # print 'Values of the cost function:'
            print 'Start to optimize the cost function...'
            
            res = optimize.fmin_cg(self._nnCostFunction, theta0, fprime=self._nnGrad, args=args, gtol=1e-3, maxiter=200)
            # gtol=1e-5, 1e-3; maxiter=None, 2, 200*6175 (default?), 1*6175, 200, 50, 100
            res = np.mat(res)
            self.theta1 = np.reshape(res[0,0:self.num_hidden_nodes*401], (400+1, self.num_hidden_nodes)).T
            self.theta2 = np.reshape(res[0,self.num_hidden_nodes*401:], (self.num_hidden_nodes+1, 10)).T
            # self.train([TrainData(self.data_matrix[i], int(self.data_labels[i])) for i in range(self.sample_num)])

            self.save()
        else:
            self._load()

    def _rand_initialize_weights(self, size_in, size_out):
        return np.mat(np.random.rand(size_out, size_in)*0.24-0.12)
        # return np.mat(np.random.rand(size_out, size_in)*0.12-0.06)

    # The sigmoid activation function. Operates on scalars.
    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)

    def _sigmoid_prime_scalar(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def draw(self, sample):
        # sample is a list? with 20*WIDTH_IN_PIXELS pixels for the hand-written digit
        pixelArray = [sample[j:j+self.WIDTH_IN_PIXELS] for j in xrange(0, len(sample), self.WIDTH_IN_PIXELS)]
        # xrange(start, stop[, step]), list comprehension, a list of lists, 2D list
        plt.imshow(zip(*pixelArray), cmap = cm.Greys_r, interpolation="nearest")
        # imshow is used to plot the image
        # zip returns a list of tuples, an array_like list, the grayscale (not a colormap)
        # One common place that interpolation happens is when you resize an image
        plt.show()
        # show is used to show the plot

    def _nnCostFunction(self, the_thetas, *args):
        
        the_thetas = np.mat(the_thetas)
        theta1 = np.reshape(the_thetas[0,0:self.num_hidden_nodes*401], (400+1, self.num_hidden_nodes)).T
        theta2 = np.reshape(the_thetas[0,self.num_hidden_nodes*401:], (self.num_hidden_nodes+1, 10)).T
        training_data_array = args
        
        J=0        
        for data in training_data_array:
            a1 = np.mat(data.fig).T
            # 400*1 matrix
            z2 = np.dot(theta1, np.row_stack((1, a1)))
            # num_hidden_nodes*1 matrix
            a2 = self.sigmoid(z2)

            z3 = np.dot(theta2, np.row_stack((1, a2)))
            # 10*1 matrix
            a3 = self.sigmoid(z3)

            y = [0] * 10 # y is a python list for easy initialization and is later turned into an np matrix (3 lines down).
            y[data.label] = 1
            # 1*10 list          
            
            for j in range(10):
                J = J + np.mat(y).T[j,0]*math.log(a3[j,0])+(1-np.mat(y).T[j,0])*math.log(1-a3[j,0])
                # numerically a3[j,0] could be smaller than 0 or larger than 1 a bit

        J = -J/self.sample_num + self.LAMBDA/(2*self.sample_num)*(np.multiply(theta1[:,1:], theta1[:,1:]).sum()+np.multiply(theta2[:,1:], theta2[:,1:]).sum())
        # print J 
        
        return J
        
    def _nnGrad(self, the_thetas, *args):
        
        the_thetas = np.mat(the_thetas)
        theta1 = np.reshape(the_thetas[0,0:self.num_hidden_nodes*401], (400+1, self.num_hidden_nodes)).T
        theta2 = np.reshape(the_thetas[0,self.num_hidden_nodes*401:], (self.num_hidden_nodes+1, 10)).T
        training_data_array = args
        
        Delta1 = np.mat(np.zeros(theta1.shape))
        # num_hidden_nodes*401 matrix
        Delta2 = np.mat(np.zeros(theta2.shape))
        # 10*(num_hidden_nodes+1) matrix
        theta1_grad = np.mat(np.zeros(theta1.shape))
        theta2_grad = np.mat(np.zeros(theta2.shape))
        
        for data in training_data_array:
            # Step 2: Forward propagation
            a1 = np.mat(data.fig).T
            # 400*1 matrix
            z2 = np.dot(theta1, np.row_stack((1, a1)))
            # num_hidden_nodes*1 matrix
            a2 = self.sigmoid(z2)

            z3 = np.dot(theta2, np.row_stack((1, a2)))
            # 10*1 matrix
            a3 = self.sigmoid(z3)
            
            # Step 3: Back propagation
            y = [0] * 10 # y is a python list for easy initialization and is later turned into an np matrix (2 lines down).
            y[data.label] = 1
            # 1*10 list
                      
            delta3 = a3 - np.mat(y).T
            # 10*1 matrix
            z2plus = np.row_stack((0, z2))
            # (num_hidden_nodes+1)*1 matrix
            delta2 = np.multiply(np.dot(theta2.T, delta3), self.sigmoid_prime(z2plus))
            # (num_hidden_nodes+1)*1 matrix
            delta2 = delta2[1:,0]
            # num_hidden_nodes*1 matrix
                      
            # Step 4: Sum delta*a.T and calculate the derivatives
            Delta1 = Delta1 + np.dot(delta2, np.row_stack((1, a1)).T)
            Delta2 = Delta2 + np.dot(delta3, np.row_stack((1, a2)).T)
        
        theta1_grad[:,0] = Delta1[:,0]/self.sample_num
        theta2_grad[:,0] = Delta2[:,0]/self.sample_num
        theta1_grad[:,1:] = Delta1[:,1:]/self.sample_num + self.LAMBDA/self.sample_num*theta1[:,1:]
        theta2_grad[:,1:] = Delta2[:,1:]/self.sample_num + self.LAMBDA/self.sample_num*theta2[:,1:] 
        
        ret = tuple(np.row_stack((theta1_grad.flatten(1).T, theta2_grad.flatten(1).T)).T.tolist()[0])
        return np.asarray(ret)

    def train(self, training_data_array): 
        data = training_data_array[0] #dict
            # Step 2: Forward propagation
        a1 = np.mat(data['y0']).T
            # 400*1 matrix
        z2 = np.dot(self.theta1, np.row_stack((1, a1)))
            # num_hidden_nodes*1 matrix
        a2 = self.sigmoid(z2)

        z3 = np.dot(self.theta2, np.row_stack((1, a2)))
            # 10*1 matrix
        a3 = self.sigmoid(z3)

            # Step 3: Back propagation
        y = [0] * 10 # y is a python list for easy initialization and is later turned into an np matrix (2 lines down).
        y[data['label']] = 1
            # 1*10 list
                      
        delta3 = a3 - np.mat(y).T
            # 10*1 matrix
        z2plus = np.row_stack((0, z2))
            # (num_hidden_nodes+1)*1 matrix
        delta2 = np.multiply(np.dot(self.theta2.T, delta3), self.sigmoid_prime(z2plus))
            # (num_hidden_nodes+1)*1 matrix
        delta2 = delta2[1:,0]
            # num_hidden_nodes*1 matrix

            # Step 4: Update weights
        self.theta1 -= self.LEARNING_RATE * np.dot(delta2, np.row_stack((1, a1)).T)
        self.theta2 -= self.LEARNING_RATE * np.dot(delta3, np.row_stack((1, a2)).T)

    def predict(self, test):
        a1 = np.mat(test).T
        # 400*1 matrix
        z2 = np.dot(self.theta1, np.row_stack((1, a1)))
        # num_hidden_nodes*1 matrix
        a2 = self.sigmoid(z2)

        z3 = np.dot(self.theta2, np.row_stack((1, a2)))
        # 10*1 matrix
        a3 = self.sigmoid(z3)        

        results = a3.T.tolist()[0]
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1":self.theta1.flatten(1).tolist()[0],
            "theta2":self.theta2.flatten(1).tolist()[0]
        };
        with open(HdrNeuralNetwork.NN_FILE_PATH,'w') as nnFile:
            json.dump(json_neural_network, nnFile)
            
        print 'nn.json is now saved'

    def _load(self):
        if not self._use_file:
            return

        with open(HdrNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = np.reshape(np.mat(nn['theta1']), (400+1, self.num_hidden_nodes)).T 
        self.theta2 = np.reshape(np.mat(nn['theta2']), (self.num_hidden_nodes+1, 10)).T 
        
        print 'reloading previous nn.json'

