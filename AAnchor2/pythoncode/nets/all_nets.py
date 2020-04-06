import os
import sys
import time
import numpy as np
import random
import glob
import threading
import timeit




import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout3D
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from keras.utils import to_categorical
from keras.models import model_from_json
from keras import regularizers

class V5_no_reg(object):
    def __init__(self):
        return
    def get_compiled_net(self):
        OPTIMIZER = 'adam'
        #define network
        ## the network ## the network
        model = Sequential()
        model.add(Conv3D(50, (3,3,3),input_shape=(11,11,11,1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Conv3D(50, (2,2,2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.50))
        model.add(MaxPooling3D(pool_size=(2, 2,2)))

        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.50))

        model.add(Dense(21))
        model.add(Activation('softmax'))

        # load weights into new model
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['categorical_accuracy'])

        return model




def get_v4():

    OPTIMIZER = 'adagrad'
    ## the network ## the network
    model = Sequential()
    model.add(Conv3D(40, (3,3,3),input_shape=(11,11,11,1)))
    model.add(Activation('relu'))
    model.add(Conv3D(40, (3,3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2,2)))

    model.add(Flatten())
    model.add(Dense(50))

    model.add(Activation('relu'))
    model.add(Dense(21))
    model.add(Activation('softmax'))

    # load weights into new model
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['accuracy'])

    return model


class V5_Drop_Reg(object):
    def __init__(self):
        return
    def get_compiled_net(self):
        OPTIMIZER = 'adam'
        #define network
        ## the network ## the network
        model = Sequential()
        model.add(Conv3D(50, (3,3,3),input_shape=(11,11,11,1), kernel_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Conv3D(50, (2,2,2),kernel_regularizer=regularizers.l2(0.1),                activity_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(0.50))
        model.add(MaxPooling3D(pool_size=(2, 2,2)))

        model.add(Flatten())
        model.add(Dense(100,kernel_regularizer=regularizers.l2(0.001),                activity_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(0.50))

        model.add(Dense(21,kernel_regularizer=regularizers.l2(0.001),                activity_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('softmax'))

        # load weights into new model
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['categorical_accuracy'])

        return model

class V5_Reg_3(object):
    def __init__(self):
        return
    def get_compiled_net(self):
        OPTIMIZER = 'adam'
        #define network
        ## the network ## the network
        model = Sequential()
        model.add(Conv3D(50, (3,3,3),input_shape=(11,11,11,1), kernel_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l2(0.00001), bias_regularizer=regularizers.l2(0.00001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Conv3D(50, (2,2,2),kernel_regularizer=regularizers.l2(0.1),                activity_regularizer=regularizers.l2(0.00001), bias_regularizer=regularizers.l2(0.00001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling3D(pool_size=(2, 2,2)))

        model.add(Flatten())
        model.add(Dense(100,kernel_regularizer=regularizers.l2(0.001),                activity_regularizer=regularizers.l2(0.0000001), bias_regularizer=regularizers.l2(0.000001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(21,kernel_regularizer=regularizers.l2(0.0000001),                activity_regularizer=regularizers.l2(0.0000001), bias_regularizer=regularizers.l2(0.0000001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('softmax'))

        # load weights into new model
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['categorical_accuracy'])

        return model


class V5_Drop_Reg_2(object):
    def __init__(self):
        return
    def get_compiled_net(self):
        OPTIMIZER = 'adam'
        #define network
        ## the network ## the network
        model = Sequential()
        model.add(Conv3D(50, (3,3,3),input_shape=(11,11,11,1), kernel_regularizer=regularizers.l2(0.005), bias_regularizer=regularizers.l2(0.0001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))

        model.add(Conv3D(50, (2,2,2),kernel_regularizer=regularizers.l2(0.005),  bias_regularizer=regularizers.l2(0.0001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))

        model.add(MaxPooling3D(pool_size=(2, 2,2)))

        model.add(Flatten())
        model.add(Dense(100,kernel_regularizer=regularizers.l2(0.005),   bias_regularizer=regularizers.l2(0.0001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dropout(0.50))

        model.add(Dense(21,kernel_regularizer=regularizers.l2(0.005),  bias_regularizer=regularizers.l2(0.0001),kernel_initializer='random_uniform',bias_initializer='zeros'))
        model.add(Activation('softmax'))

        # load weights into new model
        model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy',metrics=['categorical_accuracy'])

        return model

    def get_class_weights(self):
        class_weights={}
        for k in range(21):
            class_weights[k] = 1.0
        return class_weights
