import sys, time
import numpy as np
import keras
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, hp, Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    labels = np.loadtxt(open('../data/nq_label_for_cnn.csv'), delimiter=",")
    feats = np.loadtxt(open('../data/nq_feat_for_cnn.csv'), delimiter=",")
    max_data = np.max(feats)
    min_data = np.min(feats)
    max_label = np.max(labels)
    min_label = np.min(labels)
    feats = (feats - min_data) / (max_data - min_data)
    labels = (labels - min_label) / (max_label - min_label)
    height = 10
    width = 10
    label_size = 1
    data = np.column_stack((labels, feats))
    labels = data[:, 0:label_size]
    feats = data[:, labels.shape[1]:]
    train_num = int(0.7*labels.shape[0])
    val_num = int(0.2*labels.shape[0])
    train_feats = feats[:train_num, :] 
    val_feats = feats[train_num:train_num+val_num, :] 
    test_feats = feats[train_num+val_num:, :] 
    train_feats = train_feats.reshape(train_num, height, width, 1)
    val_feats = val_feats.reshape(val_num, height, width, 1)
    test_feats = test_feats.reshape(test_feats.shape[0], height, width, 1)
    train_labels = labels[:train_num, :] 
    val_labels = labels[train_num:train_num+val_num, :] 
    test_labels = labels[train_num+val_num:, :] 
    
    x_train = train_feats
    y_train = train_labels
    x_val = val_feats
    y_val = val_labels
    x_test = test_feats
    y_test = test_labels
    return x_train, y_train, x_val, y_val, x_test, y_test, min_label, max_label 

def build_model(x_train, y_train, x_val, y_val, x_test, y_test):
    # Model
    model = Sequential()
    model.add(Conv2D({{choice(range(4, 48, 4))}}, (3, 3), activation='relu', input_shape=(height, width, 1)))
    model.add(Conv2D({{choice(range(4, 48, 4))}}, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=1, name='bn_conv2'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D({{choice(range(8, 96, 8))}}, (3, 3), activation='relu'))
    model.add(Conv2D({{choice(range(8, 96, 8))}}, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=1, name='bn_conv4'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(15, activation='relu'))
    model.add(BatchNormalization(axis=1, name='bn_dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    
    sgd = SGD(lr={{choice([5e-4, 1e-3, 5e-3, 1e-2])}}, decay=1e-6, momentum={{choice([0.99, 0.95, 0.9])}}, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    file_path = '../models/' + 'vgg_CNN.h5'
    model.fit(x_train, y_train, 
                batch_size=32,
                validation_data=(x_val, y_val), 
                epochs={{choice(range(200, 500, 20))}})
    score = model.evaluate(x_test, y_test, verbose=0)
    try:
        with open('vgg_CNN_metric.txt') as f:
            min_error = float(f.read().strip())
    except FileNotFoundError:
        min_error = score
    if score <= min_error:
        model.save(file_path)
        with open('vgg_CNN_metric.txt', 'w') as f: 
            f.write(str(min_error))
    sys.stdout.flush()
    return {'loss': score, 'model': model, 'status': STATUS_OK}

if __name__ == '__main__':
    height = 10
    width = 25
    label_size = 5
    start = time.clock()
    best_run, best_model = optim.minimize(model=build_model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
    end = time.clock()
    print('Training time: %s Seconds' %(end - start))
    x_train, y_train, x_val, y_val, x_test, y_test, min_label, max_label = data()
    best_model = load_model('./models/vgg_CNN.h5')
    y_test = y_test*(max_label-min_label)+min_label
    start = time.clock()
    pred_test = best_model.predict(x_test)
    end = time.clock()
    print('Testing time: %s Seconds' %(end - start))
    pred_test = pred_test*(max_label-min_label)+min_label
    print("MAE:", mean_absolute_error(pred_test, y_test))
    print("RMSE:", np.sqrt(mean_squared_error(pred_test, y_test)))
    print("MAPE:", np.mean(np.abs(pred_test-y_test)/np.abs(y_test)))
    
    #extract features
    pred_train = best_model.predict(x_train)
    pred_val = best_model.predict(x_val)
    pred_test = best_model.predict(x_test)
    
    bn_dense_layer_model=Model(inputs=best_model.input, \
        outputs=best_model.get_layer('bn_dense1').output)
    feat_train=bn_dense_layer_model.predict(x_train)
    feat_val=bn_dense_layer_model.predict(x_val)
    feat_test=bn_dense_layer_model.predict(x_test)
    y_train=y_train*(max_label-min_label)+min_label
    y_val=y_val*(max_label-min_label)+min_label
    train_samples=np.c_[y_train, pred_train, feat_train, x_train.reshape(y_train.shape[0],height*width)]
    val_samples=np.c_[y_val, pred_val, feat_val, x_val.reshape(y_val.shape[0],height*width)]
    test_samples=np.c_[y_test, pred_test, feat_test, x_test.reshape(y_test.shape[0],height*width)]
    np.savetxt('../data/nq_train_samples.csv', train_samples, delimiter=",")
    np.savetxt('../data/nq_val_samples.csv', val_samples, delimiter=",")
    np.savetxt('../data/nq_test_samples.csv', test_samples, delimiter=",")
