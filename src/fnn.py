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

seed(10)

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

train_samples=np.loadtxt(open('../data/nq_train_samples.csv'), delimiter=',')
test_samples=np.loadtxt(open('../data/nq_test_samples.csv'), delimiter=',')

x_train=train_samples[:,1:]
x_test=test_samples[:,1:]

y_train=train_samples[:,0]
y_test=test_samples[:,0]
max_label = np.max(np.r_[y_train, y_test])
min_label = np.min(np.r_[y_train, y_test])
y_train=(y_train-min_label)/(max_label-min_label)
y_test=(y_test-min_label)/(max_label-min_label)

inputs=Input(shape=x_train.shape[1:], dtype='float32')
out1=Dense(1,activation='relu')(inputs)
out2=MyLayer(output_dim=32)(inputs)
out=Lambda(K.concatenate, name='concat1')([out1, out2])
out=BatchNormalization(name='bn1')(out)

out1=Dense(1,activation='relu')(out)
out2=MyLayer(output_dim=16)(out)
out=Lambda(K.concatenate, name='concat2')([out1, out2])
out=BatchNormalization(name='bn2')(out)

outputs=Dense(1, activation='sigmoid')(out)

model=Model(input=inputs,output=outputs)
model.summary()

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=0)

pred_test = model.predict(x_test)
pred_test=pred_test.reshape(pred_test.shape[0])
y_test1 = y_test*(max_label-min_label)+min_label
pred_test = pred_test*(max_label-min_label)+min_label
print "MAE:", mean_absolute_error(pred_test, y_test1), 
print "RMSE:", np.sqrt(mean_squared_error(pred_test, y_test1))
print "MAPE:", np.mean(np.abs(pred_test-y_test1)/np.abs(y_test1))
