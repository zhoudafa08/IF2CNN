from __future__ import division
import numpy as np
import keras, sys
from keras import backend as K
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Multiply, Subtract, Lambda
from numpy.random import seed
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #a = keras.layers.multiply(self.kernel)
        a = self.kernel ** 2
        #b = keras.layers.multiply(x)
        b = x ** 2
        c = K.dot(x, self.kernel)
        d = K.dot(b,a)
        #e = keras.layers.multiply(c)
        e = c ** 2
        return e-d

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

