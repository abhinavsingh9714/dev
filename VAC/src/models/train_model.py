import keras
from keras import models
from keras import layers
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import models

def build_convnet(conv_units_1,conv_units_2,conv_units_3,Dropout_rate1,lstm_units,Dropout_rate2):
    model_cnlst = models.Sequential()
    model_cnlst.add(layers.TimeDistributed(layers.Conv2D(conv_units_1, (3, 3), activation='relu', strides=(1,1)), input_shape=(10, 80, 80, 1)))
    model_cnlst.add(layers.TimeDistributed(layers.BatchNormalization()))
    model_cnlst.add(layers.TimeDistributed(layers.MaxPooling2D(2, 2)))

    model_cnlst.add(layers.TimeDistributed(layers.Conv2D(conv_units_2, (3, 3), activation='relu', padding='same')))
    model_cnlst.add(layers.TimeDistributed(layers.BatchNormalization()))
    model_cnlst.add(layers.TimeDistributed(layers.MaxPooling2D(2, 2)))

    model_cnlst.add(layers.TimeDistributed(layers.Conv2D(conv_units_3, (3, 3), activation='relu', padding='same')))
    model_cnlst.add(layers.TimeDistributed(layers.BatchNormalization()))
    model_cnlst.add(layers.TimeDistributed(layers.MaxPooling2D(2, 2)))

    # Flatten and dropout
    model_cnlst.add(layers.TimeDistributed(layers.Flatten()))
    model_cnlst.add(layers.Dropout(Dropout_rate1))

    model_cnlst.add(LSTM(lstm_units,return_sequences=False,dropout=0.2)) # used 32 units <1>

    # Dense layers
    model_cnlst.add(layers.Dense(64, activation='relu'))
    model_cnlst.add(layers.Dropout(Dropout_rate2))

    # Output layer
    model_cnlst.add(layers.Dense(6, activation='softmax'))

    print(model_cnlst.summary())
    return model_cnlst

def create_model_VGG(num_classes = 6):
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(100, 80, 3))
    vgg16.trainable = False  # Freeze VGG16 layers

    model = Sequential()
    model.add(TimeDistributed(vgg16, input_shape=(10, 100, 80, 3)))  # 10 frames per video
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(1200, return_sequences=False))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))  # num_classes should be the number of actions

    optimizers.RMSprop(learning_rate=1e-5, name="rmsprop")

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model
