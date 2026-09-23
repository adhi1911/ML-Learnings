import math 
import numpy as np 

class ActivationFunction:
    """Collection of activation funcitons and their derivatives"""

    # SIGMOID
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        sig = ActivationFunction.sigmoid(x)
        return sig * (1 - sig)
    
    # RELU
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    # TANH
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    # LINEAR
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
    
    # SOFTMAX
    @staticmethod 
    def softmax(x_list):
        max_x = max(x_list)  # For numerical stability
        exp_values = [math.exp(x - max_x) for x in x_list]
        sum_exp = sum(exp_values)
        return [exp_val / sum_exp for exp_val in exp_values]
    

    @staticmethod
    def get_activation(name):
        map ={
            'relu': ActivationFunction.relu,
            'sigmoid': ActivationFunction.sigmoid,
            'tanh': ActivationFunction.tanh,
            'linear': ActivationFunction.linear,
            'softmax': ActivationFunction.softmax
        }

        if name not in map:
            raise ValueError(f"Activation function '{name}' is not supported.")
        return map[name]
    
    @staticmethod
    def get_activation_derivative(name):
        map ={
            'relu': ActivationFunction.relu_derivative,
            'sigmoid': ActivationFunction.sigmoid_derivative,
            'tanh': ActivationFunction.tanh_derivative,
            'linear': ActivationFunction.linear_derivative
        }

        if name not in map:
            raise ValueError(f"Activation function derivative for '{name}' is not supported.")
        return map[name]