from data_utils import prepare_dataframes
from model_utils import create_data_generators, train_dense_nn_regression

import logging
import os

import tensorflow as tf

#Logging to make sure GPUs are available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
print("List available GPUs: ", gpus)
print('List available GPUs: ', tf.config.list_physical_devices('GPU'))

# Prepare the train, test and val dataframes
train, val, test = prepare_dataframes()

# Create data generators
age_train_seq, age_val_seq = create_data_generators(train, val, "age")

# Train age model, save callback weights, plot loss history.
model = train_dense_nn_regression(training_sequence = age_train_seq,
                        validation_sequence = age_val_seq,
                        callbacks_name = "callback_weights_",
                        plot_name = "../images/loss_plot.png")