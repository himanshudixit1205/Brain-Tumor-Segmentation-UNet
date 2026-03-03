# Imports

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision

import math

from sklearn.model_selection import train_test_split

from utils import *
from unet import *

import pprint # Printing objects

# YAML Configuration
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load Config Values
SEED = config["training"]["seed"]
EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
SMOOTH = config["training"]["smooth"] 
LEARNING_RATE = config["training"]["learning_rate"]
im_width = config["training"]["image_size"]
im_height = config["training"]["image_size"]

# Set seeds
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Display parameters (optional but professional)
print("Loaded Config:")
print(config)

# Seed
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# HyperParameters
LEARNING_RATE = 5e-5

# Number of Rows and Columns for Printing the image
row = 3
column = 3
n = 3


def main():
    # Loading Image Path and Mask Path
    image_filenames_train, mask_files = load_image_filename_train()

    # Plot Image and Masks
    plot_from_img_path(row, column, image_filenames_train, mask_files) # Prints Image and Mask while overlapping each other.
    show_img_mask_rows(n, image_filenames_train, mask_files) # Prints Image and Mask Image at side by side.

    # DataFrame
    df = load_pair(image_filenames_train, mask_files)

    # Train Test Split
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=SEED)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=SEED)

    # Generator 

    tf.keras.backend.clear_session() # Clear Session

    train_generator_param = config["augmentation"]

    train_gen = train_generator(
        df_train,                    # DataFrame containing training image paths and labels
        BATCH_SIZE,                 # Number of samples per training batch
        train_generator_param,      # Augmentation configuration dictionary
        target_size=(im_height, im_width)  # Resize images to model input size
    )

    # Not applying Augmentation on Val Set.
    val_gen = train_generator(
        df_val,
        BATCH_SIZE,
        dict(),  # No augmentation
        target_size=(im_height, im_width)
    )



    # Mixed precision (FASTER GPU)
    mixed_precision.set_global_policy('mixed_float16')

    # Adam Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Model
    model = unet(input_size=(im_height, im_width, 3)) # Input shape (Height, Width, RGB channels)

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=bce_dice_loss,
        metrics=['binary_accuracy', iou, dice_coefficients]
    )

    # Callbacks
    reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=config["callbacks"]["reduce_lr_factor"],
    patience=config["callbacks"]["reduce_lr_patience"],
    min_lr=config["callbacks"]["min_lr"],
    verbose=1
)

    callbacks = [
        ModelCheckpoint(
            'best_loss.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_dice.keras',
            monitor='val_dice_coefficients',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            patience=config["callbacks"]["early_stopping_patience"],
            restore_best_weights=True
        ),        
        reduce_lr
    ]

    # Safe Steps
    steps_per_epoch = math.ceil(len(df_train) / BATCH_SIZE)
    val_steps = math.ceil(len(df_val) / BATCH_SIZE)

    # Fit
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks
    )


    # Preety Print Objects
    pprint.pprint(history.history)

    # Plot Accuracy and Loss
    plot_accuracy_loss(history)



    # Load Previouly Trained Model

    model = load_model(
        "best_dice.keras",
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coefficients_loss": dice_coefficients_loss,
            "iou": iou,
            "dice_coefficients": dice_coefficients
        }
    )

    test_gen = train_generator(df_test, BATCH_SIZE, dict(), target_size = (im_height, im_width))
    results = model.evaluate(test_gen, steps=len(df_test)//BATCH_SIZE)

    print("===== TEST RESULTS =====")
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy : {results[1]:.4f}")
    print(f"IoU : {results[2]:.4f}")
    print(f"Dice: {results[3]:.4f}")


    # Plot Prediction
    for i in range(20):
        index = np.random.randint(1, len(df_test.index))
        img = cv2.imread(df_test['image_filenames_train'].iloc[index]) # Original Image NOt Mask
        img = cv2.resize(img, (im_height, im_width))
        img = img / 255
        #print(img.shape) (256, 256, 3)
        img = img[np.newaxis, :, :, :] # 3d array will become 4d array
        #print(img.shape) (1, 256, 256, 3)
        predicted_img = model.predict(img)

        plt.figure(figsize=(12,12))
        # 3 columns original image, mask, predicted image
        plt.subplot(1,3,1)
        plt.imshow(np.squeeze(img))
        plt.title('Original IMage')

        plt.subplot(1,3,2)
        plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
        plt.title("Mask Image")

        plt.subplot(1,3,3)
        plt.imshow(np.squeeze(predicted_img) > 0.3) # Checking Probabilities
        plt.title('Predicted Image')

        plt.show()
        

if __name__ == "__main__":
    main()