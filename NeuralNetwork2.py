#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:50:14 2019

@author: rishi
"""
import math, random
import numpy as np
# from matplotlib import animation

class ANN2:
    
    def __init__(self, input_dim=None, hidden_layers=None, output_dim=None, seed=1, weights_range=0.2):
        if (input_dim is None) or (output_dim is None) or (hidden_layers is None):
            raise Exception("Invalid arguments given!")
        self.input_dim = input_dim # number of input nodes
        self.output_dim = output_dim # number of output nodes
        self.hidden_layers = hidden_layers # number of hidden nodes @ each layer
        self.network = self._build_network(seed=seed,weights_range=weights_range)

    def _build_network(self, seed=1, weights_range=0.2):
        random.seed(seed)
        # Create a single fully-connected layer
        def _layer(input_dim, output_dim,weights_range):
            layer = []            
            for i in range(output_dim):
                weights = [random.random()*2*weights_range-weights_range for _ in range(input_dim)] 
                bias_weight = random.random()*2*weights_range-weights_range
                node = {"weights": weights, # list of weights
                        "bias_weight": bias_weight,
                        "v":None,# z = weights*x + bias
                        "output": None, # scalar
                        "delta": None} # scalar
                layer.append(node)
            return layer
    
        network = []
        if len(self.hidden_layers) == 0:
            network.append(_layer(self.input_dim, self.output_dim, weights_range))
        else:
            network.append(_layer(self.input_dim, self.hidden_layers[0], weights_range))
            for i in range(1, len(self.hidden_layers)):
                network.append(_layer(self.hidden_layers[i-1], self.hidden_layers[i], weights_range))
            network.append(_layer(self.hidden_layers[-1], self.output_dim, weights_range))
        return network    
        
    def fit(self, X, y, split=False, test_size=0.3, learning_rate=0.01, epochs=200, print_results=False):
            
        self.train_loss = []
        self.test_loss = []
        self.train_epoch_error = []
        self.test_epoch_error = []

        X_train, X_test, y_train, y_test = X, X, y, y
        
        for epoch in range(epochs):
                               
            for (x_, y_) in zip(X_train, y_train):
                self._forward_pass(x_) # forward pass (update node["output"])
                self._backward_pass(y_) # backward pass error (update node["delta"])
                self._update_weights(x_, learning_rate) # update weights (update node["weight"])
                    
            self.train_loss.append(np.mean(self.train_epoch_error))
            if print_results: # and epoch%100 == 0:
                print("Epoch --- ",epoch," MSE : ",self.train_loss[-1])
            self.train_epoch_error.clear()

    def _forward_pass(self, x):
        x_in = x
        for layer in self.network:
            x_out = []
            for node in layer:
                node['v'] = np.dot(node['weights'], x_in) + node['bias_weight']
                node['output'] = self._sigmoid(node['v'])
                x_out.append(node['output'])
            x_in = x_out # set output as next input
        return x_in
    
     # Backward-pass (updates node['delta'], L2 loss is assumed)
    def _backward_pass(self, y_train):
        n_layers = len(self.network)
        errors = []
        for i in reversed(range(n_layers)): # traverse backwards
            if i == n_layers - 1:
                # Difference between logits and one-hot target
                for j, node in enumerate(self.network[i]):
                    err = y_train[j] - node['output']
                    errors.append(np.square(err))
                    node['delta'] = (1/self.output_dim) * err * self._sigmoid_derivative(node['output'])
                        
            else:
                # Weighted sum of deltas from upper layers
                for j, node in enumerate(self.network[i]):
                    err = sum([node_['weights'][j] * node_['delta'] for node_ in self.network[i+1]])
                    node['delta'] = (1/self.output_dim) * err * self._sigmoid_derivative(node['output'])
        self.train_epoch_error.append(np.sum(errors))
    
    def _update_weights(self, x, learning_rate):
        for i, layer in enumerate(self.network):
            # Grab input values
            if i == 0: inputs = x
            else: inputs = [node_['output'] for node_ in self.network[i-1]]
            # Update weights
            for node in layer:
                for j, input in enumerate(inputs):
                    # dw = - learning_rate * (error * transfer') * input
                    node['weights'][j] +=  learning_rate * node['delta'] * input
                node['bias_weight'] +=  learning_rate * node['delta']

    def predict(self, X):
        ypred = np.array([self._forward_pass(x_) for x_ in X])
        return ypred
    
    def _sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x*(1.0-x)
    
    def _tanh(self,X):
        return np.tanh(X)

    def _tanh_derivative(self,X):
        return 1.0 - np.tanh(X)**2
    
    def get_adversial_image(self, X_target, y_goal, seed=0, learning_rate=0.01, _lambda=0.05, epochs=200, print_results=False):
            
        self.adversial_loss = []
        self.adversial_epoch_error = []
        self.adversial_images = []
#         x = np.copy(X_target)
        np.random.seed(seed)
        self.adversial_images.clear()
        x = np.random.randint(1,255+1, size=(self.input_dim,),)/255
        self.adversial_images.append(x)
        #print(self.adversial_images[-1])
        
        for epoch in range(epochs):
            self._forward_pass_adversial(x) # forward pass (update node["output"])
            self._backward_pass_adversial(y_goal) # backward pass error (update node["delta"])
                    
            self.adversial_loss.append(self.train_epoch_error + _lambda * np.sum(np.square(X_target-x)))
            if print_results and epoch%100 == 0:
                print("Epoch --- ",epoch," MSE : ",self.adversial_loss[-1])
            self.adversial_epoch_error.clear()
            x = self._update_inputs_adversial(x, X_target, learning_rate, _lambda) # update weights (update node["weight"])
            self.adversial_images.append(x)
            #print(self.adversial_images[-1])
        return x, self.adversial_images
    
    def _forward_pass_adversial(self, x):
        x_in = x
        for layer in self.network:
            x_out = []
            for node in layer:
                node['v'] = np.dot(node['weights'], x_in) + node['bias_weight']
                node['output'] = self._sigmoid(node['v'])
                x_out.append(node['output'])
            x_in = x_out # set output as next input
    
    def _backward_pass_adversial(self, y_goal):
        n_layers = len(self.network)
        errors = []
        for i in reversed(range(n_layers)): # traverse backwards
            if i == n_layers - 1:
                # Difference between logits and one-hot target
                for j, node in enumerate(self.network[i]):
                    err = y_goal[j] - node['output']
                    errors.append(np.square(err))
                    node['delta'] = err * self._sigmoid_derivative(node['output'])
                        
            else:
                # Weighted sum of deltas from upper layers
                for j, node in enumerate(self.network[i]):
                    err = sum([node_['weights'][j] * node_['delta'] for node_ in self.network[i+1]])
                    node['delta'] = err * self._sigmoid_derivative(node['output'])
        self.adversial_epoch_error.append(np.sum(errors)/2)
    
    def _update_inputs_adversial(self, x, X_target, learning_rate, _lambda):
        for i in range(x.shape[0]):
            delta = sum([node_['weights'][i] * node_['delta'] for node_ in self.network[0]])
            x[i] += learning_rate * (delta + _lambda * (X_target[i] - x[i])) 
        return x