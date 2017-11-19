import os
import pandas as pd

from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from utils import imgs_batch_processing
import matplotlib.pyplot as plt

def load_data(paras):
    img_path = os.path.join(paras['data_dir'],'driving_log.csv')
    data = pd.read_csv(img_path, names=['center','left','right',
                                        'steering_angle', 'throttle','brake', 'spped'],
                                        header=None)
                    
    X = data[['center', 'left', 'right']].values
    y = data['steering_angle'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=paras['test_size'])
    return X_train, X_valid, y_train, y_valid

def model(paras):
    """
    NIVIDA self-driving CNN model 
    """
    model = Sequential()
    # normalized input planes
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66, 200, 3)))
    # five cnn layers
    # subsample=(1, 1),left and down stride 向左和向下的过滤窗口移动步幅
    # karas 如何设置padding呢？
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(paras['keep_prob']))
    # Flatten lyaer
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

def training_model(model, paras, X_train, X_valid, y_train, y_valid):
    model_checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only='true',
                                       mode='auto')
   
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=paras['learning_rate']))

    history = model.fit_generator(imgs_batch_processing(paras['data_dir'], X_train, y_train, paras['batch_size'], True),
                        paras['num_per_epoch'],
                        paras['epochs'],
                        max_q_size=1,
                        validation_data=imgs_batch_processing(paras['data_dir'], X_valid, y_valid, paras['batch_size'], False),
                        nb_val_samples=len(X_valid),
                        callbacks=[model_checkpoint],
                        verbose=1)
    
    plt.figure()
    plt.plot(history.history['loss'],'x-')
    plt.plot(history.history['val_loss'],'o-')
    plt.legend(['X_train', 'X_valid'], loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.savefig('model_loss.png')
    plt.show()


if __name__ == '__main__':
    paras = {}
    paras['data_dir'] = 'dataset1'
    paras['test_size'] = 0.2
    paras['epochs'] = 10
    paras['num_per_epoch'] = 13000
    paras['batch_size'] = 256
    paras['learning_rate'] = 5.0e-4
    paras['keep_prob'] = 0.5
    
    print('loading data ... ')
    X_train, X_valid, y_train, y_valid = load_data(paras)
    print('training set size:',X_train.size)
    print()
    print()
    print('*********************************************************')
    print()
    print('build nivida model with dropout ...')
    model = model(paras)
    print()
    print()
    print('*********************************************************')
    print()
    print('start training ...')
    training_model(model, paras, X_train, X_valid, y_train, y_valid)