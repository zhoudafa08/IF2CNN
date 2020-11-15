import os, time
import numpy as np
import keras, sys
from sklearn import metrics
from keras.models import load_model
from keras import backend as K
from keras import Input, Model, losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Multiply, Subtract, Lambda
from numpy.random import seed
from keras.wrappers.scikit_learn import KerasRegressor
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from MyLayer import MyLayer

seed(10)

def data():
    train_samples=np.loadtxt(open('../data/nq_train_samples.csv'), delimiter=',')
    val_samples=np.loadtxt(open('../data/nq_val_samples.csv'), delimiter=',')
    test_samples=np.loadtxt(open('../data/nq_test_samples.csv'), delimiter=',')
    feat_name=sys.argv[1]
    fc_num=int(sys.argv[2])
    cnn_feat_num=fc_num+1
    if feat_name=='if':
        x_train=train_samples[:,cnn_feat_num:]
        x_val=val_samples[:,cnn_feat_num:]
        x_test=test_samples[:,cnn_feat_num:]
    elif feat_name=='cnn':
        x_train=train_samples[:,1:cnn_feat_num]
        x_val=val_samples[:,1:cnn_feat_num]
        x_test=test_samples[:,1:cnn_feat_num]
    elif feat_name=='all':
        x_train=train_samples[:,1:]
        x_val=val_samples[:,1:]
        x_test=test_samples[:,1:]
    elif feat_name=='none':
        feats = np.loadtxt(open('../data/lod_if_feats_long_1221.csv'), delimiter=",")
        feats_all = np.sum(feats.reshape(feats.shape[0], 10, 25), axis=1)
        max_data = np.max(feats_all)
        min_data = np.min(feats_all)
        feats = (feats_all - min_data) / (max_data - min_data)
        train_num = int(0.7*feats_all.shape[0])
        val_num = int(0.2*feats_all.shape[0])
        x_train = feats[:train_num, :] 
        x_val = feats[train_num:train_num+val_num, :] 
        x_test = feats[train_num+val_num:, :]
    y_train=train_samples[:,:1]
    y_val=val_samples[:,:1]
    y_test=test_samples[:,:1]
    max_label = np.max(np.r_[y_train, y_val, y_test])
    min_label = np.min(np.r_[y_train, y_val, y_test])
    y_train=(y_train-min_label)/(max_label-min_label)
    y_test=(y_test-min_label)/(max_label-min_label)
    return x_train, y_train, x_test, y_test, max_label, min_label

def create_model(x_train, y_train, x_test, y_test, max_label, min_label):
    inputs=Input(shape=x_train.shape[1:], dtype='float32')
    out1=Dense({{choice(range(1, 10))}})(inputs)
    out2=MyLayer(output_dim={{choice(range(2, 64, 4))}})(inputs)
    out=Lambda(K.concatenate, name='concat1')([out1, out2])
    out=Activation('relu')(out)
    out=BatchNormalization(name='bn1')(out)
    out1=Dense({{choice(range(1, 10))}})(out)
    out2=MyLayer(output_dim={{choice(range(4, 128, 4))}})(out)
    out=Lambda(K.concatenate, name='concat2')([out1, out2])
    out=Activation('relu')(out)
    out=BatchNormalization(name='bn2')(out)
    outputs=Dense(1, activation='sigmoid')(out)
    
    model=Model(input=inputs,output=outputs)
    model.summary()
    sgd = SGD(lr={{choice([5e-4, 1e-3, 5e-3, 1e-2])}}, decay=1e-6, momentum={{choice([0.99, 0.95, 0.9])}}, nesterov=True)
    model.compile(loss=losses.mean_squared_error,  optimizer=sgd, metrics=['mse'])
    model.fit(x_train, y_train, batch_size=32, epochs={{choice(range(100, 800, 20))}}, verbose=0, validation_data=(x_val, y_val))
    pred = model.predict(x_test)
    score = mean_squared_error(pred, y_test)
    file_path = '../models/fnn_model.h5'
    try:
        with open('fnn_metric.txt') as f:
            min_error = float(f.read().strip())
    except FileNotFoundError:
        min_error = score
    if score <= min_error:
        model.save(file_path)
        with open('fnn_metric.txt', 'w') as f: 
            f.write(str(min_error))
    sys.stdout.flush()
    return {'loss': score, 'model': model, 'status': STATUS_OK}

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, max_label, min_label = data()
    start = time.clock()
    best_run, best_model = optim.minimize(model=create_model,\
                                          data=data, \
					  algo=tpe.suggest,\
					  max_evals=25,\
					  trials=Trials())
    end = time.clock()
    print('Training time: %s Seconds' %(end - start))
    best_model = load_model('./models/fnn_model.h5', custom_objects={'MyLayer':MyLayer})
    start = time.clock()
    pred_test = best_model.predict(x_test)
    end = time.clock()
    print('Testing time: %s Seconds' %(end - start))
    pred_test=pred_test.reshape(y_test.shape)
    y_test = y_test*(max_label-min_label)+min_label
    pred_test = pred_test*(max_label-min_label)+min_label
    print("MAE:", mean_absolute_error(pred_test, y_test))
    print("RMSE:", np.sqrt(mean_squared_error(pred_test, y_test)))
    print("MAPE:", np.mean(np.abs(pred_test-y_test)/np.abs(y_test)))

