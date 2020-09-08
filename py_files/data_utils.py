import numpy as np
import pandas as pd
from PIL import Image


# Prepare the train, val, and test dataframes.
# Used in training, prediction, and visualization.
def prepare_dataframes():
    # Data files from UKBB
    filenames_file = "../data/left_eye_filenames.txt"
    master_phenotypes_file = "../data/trimmed_master.txt"
    fundus_phenotypes_file = "../data/fundus_phenotypes.csv"

    # Parse the filenames into a filenames_df, filter to only "0_0" files.
    ff = open(filenames_file, "r")
    filenames = np.array(ff.readlines())
    filenames = [i.rsplit() for i in filenames]
    filenames = [j.split("_") for i in filenames for j in i]
    filenames = [i for i in filenames if i[2] == "0" and i[3] == "0"]
    filenames_df = pd.DataFrame(filenames, columns=["EID", "which_eye", "2", "3", "4"], dtype=str)
    
    # Create master_df
    master_df = pd.read_csv(master_phenotypes_file, sep=' ', dtype=str)
    master_df = master_df[:-1]

    # Create fundus_df
    fundus_df = pd.read_csv(fundus_phenotypes_file, dtype=str)
    fundus_df = fundus_df[:-1]

    # Merge all dfs
    merged = pd.merge(filenames_df, master_df, how="left", left_on="EID", right_on="FID")
    merged = pd.merge(merged, fundus_df, how="left", left_on="EID", right_on="EID")

    # Select the columns of interest, specify data types.
    merged = merged[merged["sex"].isin(["0", "1"])]
    merged = merged.astype({'sex': int, "ass_1_age": float})
    merged = merged[["EID", "population", "split", "ass_1_age", "sex"]]
    merged = merged.rename(columns={"ass_1_age": "age"})

    # Recreate filenames
    # Might have to change this path depending on where you clone the imaging repo:
    merged["filenames"] = "../../../../../../[redacted]" + merged["EID"].astype(str) + "_21015_0_0_resized.png" #This file path has been removed to ensure the privacy of Rivas Lab.
    
    # Split into train, val and test. This will automatically filter to population="white_british".
    train = merged[merged["split"] == "train"]
    val = merged[merged["split"] == "val"]
    test = merged[merged["split"] == "test"]

    return train, val, test

# Crop to the area around the optic disc.
# Used in preprocessing for training, prediction, and visualization.
def crop(image):
    return image[175:375, 15:200, :]

# Returns the shape of the crop function.
def cropped_shape():
    return crop(np.ones((587, 587, 3))).shape

# Mean and standard deviation by channel for normalization.
# Used in preprocessing for training, prediction, and visualization.
def precomputed_mean_sd():
    return np.array([183, 95, 36]), np.array([65, 43, 24])

# Calculate the mean and standard deviation by channel.
# Use this only when running the model on NEW data.
# Plug the output into precomputed_mean_sd().
def compute_mean_sd(train, prop=0.1):
    imgs = [crop(np.asarray(Image.open(row["filenames"]))) for index, row in train.sample(frac=prop).iterrows()]
    imgs_array = np.stack(imgs, axis=0)
    return imgs_array.mean(axis=(0, 1, 2)), imgs_array.std(axis=(0, 1, 2))
