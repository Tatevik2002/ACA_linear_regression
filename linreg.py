import numpy as np
import pandas as pd
class LinearRegression:
    def __init__(self, x,y , method, lr=0.1):
        """
        :param method: 'sgd' for SGD, 'gd' for GD, 'analytic' for analytic solution
        "param lr: learning rate
        """
        self._lr = lr

        if method not in ('analytic', 'gd', 'sgd'):
            raise ValueError('method can be only "gd", "sgd" or "analytic"')

        self._method = method
        self._x = None
        self._y = None
        
        

    def loss_function(self):
        """
        TODO: calculate the loss function
        """
        if self._x is None or self._y is None:
            raise ValueError('All methods can be called after fit method is called.')

        return np.transpose(self._y-self._x@self._w)@(self._y-self._x@self._w)
        

    def gradient(self, x, y):
        """
        Calculate the gradient of the loss function.
        If x is a vector, calculate only for this data (for SGD), else for whole dataset (for GD)
        """
        grad = -2*self._x@self._y+2*np.transpose(self._x)@self._x@self._w

        



    def fit(self, x, y):
        """
        TODO: normalize the data and fit the linear regression.
        :param x: features matrix
        :param y: labels
        :returns: None if can't fit, weights, if fitted.
        """
        x_ = []
        y_ = []
        self._x = x_
        self._y = y_
        for i in range(len(self._x)):
            x_.append(self._x[i])
        for i in range(len(self._y)):
            y_.append(self._y[i])
        max_x = max(x_)
        max_y = max(y_)
        for i in range(len(self._x)):
            self._x[i]/= max_x
        for i in range(len(self._y)):
            self._y[i]/= max_y

        w = np.zeros(shape=(len(x),))
        self._w = w
        

        if self._method == 'sgd':
            pass
            for i in len(self._x):
                self._x= self._x[i,i+50]
                self._y= self._y[i,i+50]
                self._w -= self._lr*self.gradient(self._x,self._y)
                if self.loss_function()<=100:
                    return self._w
            return self._w
            
        elif self._method == 'gd':
            while self.loss_function()>100:
                self._w -= self._lr*self.gradient(self._x,self._y)
        elif self._method == 'analytic':
            self._w = np.linalg.inv(np.transpose(self._x)@self._x)@np.transpose(self._x)@self._y
        else:
            raise ValueError('method can be only "gd", "sgd" or "analytic"')

        return self._w

    def predict(self, x):
        """
        TODO: Calculate the predictions for each data in features matrix.
        :param x: features matrix
        """
        y_pred = self._x@self.fit(self._x, self._y)
        return y_pred


