from data_utils import prepare_dataframes, precomputed_mean_sd, crop, cropped_shape
from model_utils import load_dense_nn_regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import scipy.stats

#Load Model
model = load_dense_nn_regression()

# Prepare the train, val, and test dataframes
train, val, test = prepare_dataframes()

#Trim test set
size = 6527 #Tweak this for a smaller test set. Total = 6527
test = test.head(size)

#Preprocess Images
arrs = [np.asarray(Image.open(image_path)) for image_path in test["filenames"]]
arrs = [crop(arr) for arr in arrs]
mean_array, sd_array = precomputed_mean_sd()
arrs = [(arr - mean_array)/(sd_array) for arr in arrs]
arrs = np.array(arrs)

#Predict Ages
predictions = np.array(model(arrs, training=False)).reshape(size)

#Load True Ages and Baseline
true_ages = np.array(test["age"])
mean_age = true_ages.mean()
mean_ages = np.ones(size) * mean_age

#Calculate MAE relative to baseline
baseline_MAE = np.linalg.norm(mean_ages - true_ages, ord=1)/size
print("Baseline MAE:", "{:.2f}".format(baseline_MAE))
predict_MAE = np.linalg.norm(predictions - true_ages, ord=1)/size
print("Prediction MAE:", "{:.2f}".format(predict_MAE))

#Calculate r^2
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mean_ages, true_ages)
print("Baseline r-squared: ", "{:.2f}".format(r_value**2))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(predictions, true_ages)
print("Prediction r-squared: ", "{:.2f}".format(r_value**2))

#Plot true vs predicted
fig, ax = plt.subplots()
fig.suptitle("Age Predictions - Test Set")
colors = scipy.stats.gaussian_kde(np.vstack([true_ages, predictions]))(np.vstack([true_ages, predictions]))
idx = colors.argsort()
true_ages, predictions, colors = true_ages[idx], predictions[idx], colors[idx] #Sort points so that highest density is plotted last
ax.scatter(true_ages, predictions, c=colors, s=100, edgecolors=None)
ax.plot([true_ages.min(), true_ages.max()], [true_ages.min(), true_ages.max()], 'k--', lw=4)
ax.set_xlabel('True Ages')
ax.set_ylabel('Predicted Ages')
plt.xlim(35, 75)
plt.ylim(35, 75)
plt.savefig('../images/scatter.png')
