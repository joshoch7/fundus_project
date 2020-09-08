from data_utils import prepare_dataframes, precomputed_mean_sd, crop, cropped_shape
from model_utils import load_dense_nn_regression

from alibi.explainers import IntegratedGradients

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

#Load model
model = load_dense_nn_regression()

# Prepare the train, val, and test dataframes
train, val, test = prepare_dataframes()

#Set the number of images to visualize.
size = 50

for index, row in test.head(size).iterrows():
    #Setup Subplot
    fig, (ax1, ax2) = plt.subplots(1, 2)

    #Preprocess, and print image #1
    arr = crop(np.asarray(Image.open(row["filenames"])))
    ax1.imshow(arr)
    mean_array, sd_array = precomputed_mean_sd()
    arr = (arr - mean_array)/(sd_array)
    arr = arr.reshape((1, *arr.shape))

    #Predict age
    prediction = np.array(model(arr, training=False))

    #Print image #2 - Integrated Gradients
    ig  = IntegratedGradients(model, n_steps=10, method="gausslegendre")
    explanation = ig.explain(arr, baselines=None, target=prediction)
    attrs = explanation.attributions
    attrs = np.abs(attrs.squeeze())
    im2 = ax2.imshow(attrs)

    #Hide axes
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)

    #Set title and save.
    fig.suptitle("Age: " + "{:.2f}".format(row["age"]) + "; Prediction: " + "{:.2f}".format(prediction[0][0]))
    fig.savefig("../images/explanations/explanation_" + str(index) + ".png")
