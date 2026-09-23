import math 
import numpy as np 

class LossFunction:
    """
    Collection of loss functions and their derivatives
    """

    @staticmethod 
    def mse(predictions, targets):
        """Mean Squared Error Loss"""
        squared_errors = [(pred -tgt) **2 for pred, tgt in zip(predictions, targets)]
        return np.mean(squared_errors)
    
    @staticmethod
    def mae(predictions, targets):
        """Mean Absolute Error Loss"""
        absolute_errors = [abs(pred - tgt) for pred, tgt in zip(predictions, targets)]
        return np.mean(absolute_errors)
    

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
    def get_loss_function(name):
        map = {
            'mse': LossFunction.mse,
            'mae': LossFunction.mae,
            'huber': LossFunction.huber_loss
        }
        return map.get(name, None)
    
