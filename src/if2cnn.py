from __future__ import division
import numpy as np
import keras
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sys

# Generate dummy data
labels = np.loadtxt(open('../data/nq_label_for_cnn.csv'), delimiter=",")
feats = np.loadtxt(open('../data/nq_feat_for_cnn.csv'), delimiter=",")
max_data = np.max(feats)
min_data = np.min(feats)
max_label = np.max(labels)
min_label = np.min(labels)
feats = (feats - min_data) / (max_data - min_data)
labels = (labels - min_label) / (max_label - min_label)
#print labels.shape, feats.shape

height = 10
width = 10
label_size = 1
data = np.column_stack((labels, feats))
#np.random.shuffle(data)
labels = data[:, 0:label_size]
feats = data[:, labels.shape[1]:]
train_num = int(0.8*labels.shape[0])
test_num=feats.shape[0]-train_num
train_feats = feats[:train_num, :] 
test_feats = feats[train_num:, :] 
#print train_feats.shape

train_feats = train_feats.reshape(train_num, height, width, 1)
test_feats = test_feats.reshape(test_feats.shape[0], height, width, 1)
#feats = feats.reshape((1,) + feats.shape)
train_labels = labels[:train_num, :] 
test_labels = labels[train_num:, :] 

x_train = train_feats
y_train = train_labels
x_test = test_feats
y_test = test_labels

inputs = Input(shape=x_train.shape[1:], dtype='float32')
# Model
# input: 10x10 images with 1 channels -> (10, 10, 1) tensors.
# this applies 32 convolution filters of size 3x3 each.
out=Conv2D(32, (3, 3), name='conv1')(inputs)
out=Activation('relu')(out)
out=Conv2D(32, (3, 3), name='conv2')(out)
out=Activation('relu')(out)
out=BatchNormalization(name='bn_conv2')(out)
out=Dropout(0.25)(out)

# this applies 64 convolution filters of size 3x3 each.
out=Conv2D(64,(3,3), name='conv3')(out)
out=Activation('relu')(out)
out=Conv2D(64,(3,3), name='conv4')(out)
out=Activation('relu')(out)
out=BatchNormalization(name='bn_conv4')(out)
out=Dropout(0.25)(out)

# this applies full connection layer of 10 neruons.
out=Flatten()(out)
out=Dense(10)(out)
out=Activation('relu', name='act_dense1')(out)
out=BatchNormalization(name='bn_dense1')(out)
out=Dropout(0.5)(out)

# this applies full connection layer of 1 output neruons.
outputs=Dense(1, activation='sigmoid', name='dense_2')(out)

model=Model(input=inputs,output=outputs)
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=400)

y_test = y_test*(max_label-min_label)+min_label
pred_test = model.predict(x_test)
pred_test = pred_test*(max_label-min_label)+min_label
print "MAE:", mean_absolute_error(pred_test, y_test), 
print "RMSE:", np.sqrt(mean_squared_error(pred_test, y_test))
print "MAPE:", np.mean(np.abs(pred_test-y_test)/np.abs(y_test))
pred_train = model.predict(x_train)
pred_test = model.predict(x_test)

bn_dense_layer_model=Model(inputs=model.input, \
    outputs=model.get_layer('bn_dense1').output)
feat_train=bn_dense_layer_model.predict(x_train)
feat_test=bn_dense_layer_model.predict(x_test)
y_train=y_train*(max_label-min_label)+min_label
train_samples=np.c_[y_train, pred_train, feat_train, x_train.reshape(train_num,height*width)]
test_samples=np.c_[y_test, pred_test, feat_test, x_test.reshape(test_num,height*width)]
np.savetxt('../data/nq_train_samples.csv', train_samples, delimiter=",")
np.savetxt('../data/nq_test_samples.csv', test_samples, delimiter=",")
