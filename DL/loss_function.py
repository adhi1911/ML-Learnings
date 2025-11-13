import math 
import numpy as np 

class LossFunction:
    """
    Collection of loss functions and their derivatives
    """

    @staticmethod 
    def mse(predictions, targets):
        """Mean Squared Error Loss"""
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def mae(predictions, targets):
        """Mean Absolute Error Loss"""
        return np.mean(np.abs(predictions - targets))
    

    @staticmethod
    def huber_loss(predictions, target):
        """Huber Loss"""
        delta = 1.0
        error = predictions - target
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * (error ** 2)
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    

    @staticmethod 
    def loss(name):
        map = {
            'mse': LossFunction.mse,
            'mae': LossFunction.mae,
            'huber': LossFunction.huber_loss
        }
        return map.get(name, None)
    
