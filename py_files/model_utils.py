from data_utils import precomputed_mean_sd, crop, cropped_shape

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D,
                                     GlobalAveragePooling2D,
                                     BatchNormalization,
                                     Flatten, Dropout, Dense, ReLU,
                                     AveragePooling2D, Concatenate, Lambda)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence


# Helper class for create_data_generators()
class ImageGen(Sequence):

    # Initializer
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    # Return number of batches
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    # Create a batch
    def __getitem__(self, idx):
        #Get batch
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #Preprocess images
        arrs = [np.asarray(Image.open(image_path)) for image_path in batch_x]
        arrs = [crop(arr) for arr in arrs]
        mean_array, sd_array = precomputed_mean_sd()
        arrs = [(arr - mean_array)/(sd_array) for arr in arrs]

        return np.array(arrs), np.array(batch_y)

# Create data generators for training.
def create_data_generators(train, val, label_type):
    training_sequence = ImageGen(list(train["filenames"]), list(train[label_type]), 32)
    validation_sequence = ImageGen(list(val["filenames"]), list(val[label_type]), 32)
    return training_sequence, validation_sequence
    

# Citation for Densenet implementation: https://github.com/taki0112/Densenet-Tensorflow
#######################################################################################
n_filters = 12

def conv_layer(x, filter, kernel, strides=1, name="conv"):
    x = Conv2D(
        filters=filter, kernel_size=kernel, strides=strides,
        padding='same',use_bias=False,name=name+'_conv',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution="truncated_normal"))(x)
    return x

def bottleneck_layer(x, filters, name, dropout_rate=0.2):
    x = BatchNormalization(name=name+'_bn_1',axis=-1)(x)
    x = ReLU()(x)
    x = conv_layer(x, filter=4*filters, kernel=1, name=name+'_conv1')
    x = Dropout(dropout_rate, name=name+'_dropout_first')(x)
    x = BatchNormalization(name=name+"_bn_2")(x)
    x = ReLU()(x)
    x = conv_layer(x, filter=filters, kernel=3, name=name+'_conv2')
    x = Dropout(dropout_rate, name=name+'_dropout_second')(x)
    return x

def transition_layer(x,name):
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    in_channel = x.shape[-1]
    in_channel = int(int(in_channel)*0.5)
    x = conv_layer(x, filter=in_channel , kernel=1, name=name+'_conv')
    x = Dropout(0.2)(x)
    x = AveragePooling2D( pool_size=2, strides=2,padding='valid')(x)
    return x


def dense_block(input_x, nb_layers,name):
    layers_concat = []
    layers_concat.append(input_x)
    x = bottleneck_layer(input_x, filters=n_filters, name=name + '_bottleN_' + str(0))
    layers_concat.append(x)

    for i in range(nb_layers - 1):
        x = Concatenate(axis=-1)(layers_concat[:])
        x = bottleneck_layer(x, filters=n_filters, name=name + '_bottleN_' + str(i + 1))
        layers_concat.append(x)

    x = Concatenate()(layers_concat[:])
    return x

def dense_nn_regression():
    inputs = tf.keras.Input(shape=cropped_shape())
    x = conv_layer(inputs, filter=2*n_filters, kernel=7, strides=2,name='conv_1')
    x = MaxPool2D(pool_size=3, strides=2, padding='valid')(x)
    x = dense_block(input_x=x, nb_layers=6,name='dense0')
    x = transition_layer(x,name='trans1')
    x = dense_block(input_x=x, nb_layers=12,name='dense1')
    x = transition_layer(x,name='trans2')
    x = dense_block(input_x=x, nb_layers=48,name='dense3')
    x = transition_layer(x,name='trans3')
    x = dense_block(input_x=x, nb_layers=32,name='dense_final')

    # 100 Layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Flatten()(x)
    output = Dense(1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                               mode='fan_in',
                                                                               distribution="truncated_normal"))(x)

    return tf.keras.Model(inputs=inputs, outputs=output, name='dense_nn_regression')
#######################################################################################

# Train model, save callback weights, plot loss history.
def train_dense_nn_regression(training_sequence, validation_sequence, callbacks_name, plot_name):
    # Create model
    model = dense_nn_regression()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, epsilon=0.1),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    # Setup callbacks
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="../models/" + callbacks_name + "{epoch:02d}-{val_loss:.2f}.h5",
        # filepath='../models/large_model1_callbacks{epoch:02d}-{val_loss:.2f}.h5',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Train model
    history = model.fit(x=training_sequence, validation_data=validation_sequence, epochs=100, callbacks=[model_checkpoint_callback], workers=8, use_multiprocessing=True)

    #Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(plot_name)

    return model

# Load a trained model
# To test, replace the callback file with the latest callback from the most recent trained model.
def load_dense_nn_regression():
    model = dense_nn_regression()
    model.load_weights("../models/callback_weights_32-31.81.h5")
    return model
