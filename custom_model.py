
import cv2, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, glob
import tensorflow as tf
import time 

from keras.models import Sequential
from keras import losses, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, CSVLogger, History, ModelCheckpoint, EarlyStopping



# NVIDIA model
def nvidia_model(img_height, img_width, img_channels):
    model = Sequential()
    model.add(Lambda(lambda x: x/255., input_shape=(img_height, img_width, img_channels)))
    # Cov layers
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
    # Fullyconnected layer
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adadelta', metrics=['mse'])
    # model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
    return model


# custom model
def custom_model(img_height, img_width, img_channels):
    model = Sequential()
    # norm
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=(img_height, img_width, img_channels)))
    # Cov 
    model.add(Conv2D(24, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='re_conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool1'))
    
    model.add(Conv2D(36, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='re_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool2'))
    
    model.add(Conv2D(48, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='re_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool3'))
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='re_conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool4'))
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='re_conv5'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='re_maxpool5'))
     
    # Fullyconnected layer
    model.add(Flatten())
    model.add(BatchNormalization(name='re_bn1'))
    model.add(Dense(1164, activation='relu', kernel_initializer='he_normal', name='re_den1'))
    model.add(Dropout(0.25))
    
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal', name='re_den2'))
    model.add(Dropout(0.25))
    
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal', name='re_den3'))
    model.add(Dropout(0.25))
    
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', name='re_den4'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, kernel_initializer='he_normal'))
    
    model.compile(loss='mse', optimizer='adadelta', metrics=['mse'])
    # model.compile(loss='mse', optimizer='Adam', metrics=['mse'])
    return model



def fine_tuning(ft_model, X_train, y_train, X_valid, y_valid, freeze_layer, batch_size=64, epochs=2, weights=None):

    for layer in ft_model.layers:
        layer.trainable = True
    for layer in ft_model.layers[:freeze_layer]:
        layer.trainable = False
        
    ft_model.compile(optimizer='Adam', loss='mse', metrics=['mse'])
    
    if weights:
        ft_model.load_weights(weights)

    logs_file = 'finetune-%s-{val_loss:.4f}.h5'%str(freeze_layer)
    path = os.getcwd()
    path_logs = os.path.join(path, logs_file)

    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    model_check = ModelCheckpoint(path_logs, monitor='val_loss', save_best_only=True)
    
    ft_model_history = ft_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, 
                                     validation_data=(X_valid, y_valid), callbacks=[early_stop, model_check])
    
    return ft_model, ft_model_history
