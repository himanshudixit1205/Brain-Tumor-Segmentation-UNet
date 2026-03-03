import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from glob import glob
from dotenv import load_dotenv
import os


# YAML Configuration
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load Dotenv File
load_dotenv()   

# Parameters
SEED = config["training"]["seed"]
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# The Dice Coefficient (also called Dice Score or Dice Similarity Coefficient - DSC) is a metric used to measure overlap between two regions.
def dice_coefficients(y_true, y_pred, smooth=1):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=1):
    return -dice_coefficients(y_true, y_pred, smooth) # Loss Metric

# Intersection over Union (IoU) also known as the Jaccard Index or Jaccard similarity coefficient, is a fundamental metric used to measure the similarity and overlap between two sets. In computer vision, it is the standard for evaluating how accurately a model localizes objects.
def iou(y_true_flatten, y_pred_flatten, smooth=1): 
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    add = K.sum(y_true_flatten + y_pred_flatten)
    iou = (intersection + smooth) / (add - intersection + smooth) # Jaccard IOU
    return iou

# Jaccard distance is a measure of dissimilarity between two sets, derived directly from Jaccard similarity. While Jaccard similarity measures how much two sets overlap, Jaccard distance measures how different they are. It is defined as one minus the Jaccard similarity.
def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten) # Loss Metric


# Combines BCE and Dice loss to improve segmentation accuracy and overlap.
def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(binary_crossentropy(y_true, y_pred))
    dice = dice_coefficients_loss(y_true, y_pred)
    return (0.2 * bce) + (0.8 * dice)
# The Loss Function is the goal, and the Optimizer is the strategy to reach it.


# Image with Mask Image
def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
    fig = plt.figure(figsize=(12,12))
    rnge = rows * columns + 1

    for i in range(1, rnge):
        fig.add_subplot(rows, columns, i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)

    plt.show()

# Image and Mask Image side by side
def show_img_mask_rows(n, list_img_path, list_mask_path):
    fig = plt.figure(figsize=(6, 3 * n))

    plot_idx = 1

    for i in range(n):
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)  # grayscale mask

        # --- IMAGE ---
        fig.add_subplot(n, 2, plot_idx)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")
        plot_idx += 1

        # --- MASK ---
        fig.add_subplot(n, 2, plot_idx)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis("off")
        plot_idx += 1

    plt.tight_layout()
    plt.show()


# After Mask Normalization value is less than or equal to 0.5. Then we will skip that mask.
def normalize_and_diagnose(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1 # Its Tumor
    mask[mask <= 0.5] = 0 # Not a Tumor
    return (img, mask)

# Referring Code from : https://github.com/zhixuhao/unet/blob/master/data.py
# Using 'train_generator' for Load and process data in batches while training, instead of loading everything into memory.
def train_generator(
    data_frame,
    batch_size,
    augmentation_dict,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
):
    """
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    """
    image_datagen = ImageDataGenerator(**augmentation_dict)
    mask_datagen = ImageDataGenerator(**augmentation_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,                    # Pandas DataFrame containing file paths
        x_col="image_filenames_train",# Column name with image file paths
        class_mode=None,              # No labels returned (augmentation only)
        color_mode=image_color_mode,  # 'rgb' or 'grayscale' image mode
        target_size=target_size,      # Resize images to (height, width)
        batch_size=batch_size,        # Images per batch
        save_to_dir=save_to_dir,      # Directory to save augmented images
        save_prefix=image_save_prefix,# Prefix for saved filenames
        seed=seed,                    # Random seed for reproducibility
        shuffle=False                 # No shuffling of images
    )


    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        shuffle=False

    )

    train_gen = zip(image_generator, mask_generator)

    # Final return Tuple after image Normalization and Diagnostics
    for (img, mask) in train_gen:
        img, mask = normalize_and_diagnose(img, mask)
        yield (img, mask)


# Load Image and Mask Files
def load_image_filename_train():
    image_filenames_train = []
    path = os.getenv("DATASET_PATH")
    mask_files = [f for f in glob(f"{path}/**/*_mask*") 
              if "Zone.Identifier" not in f]
    for i in mask_files:
        image_filenames_train.append(i.replace('_mask',''))
    print(image_filenames_train[:10])
    return image_filenames_train, mask_files


# Load Valid Pairs
def load_pair(image_filenames_train, mask_files):
    # Check if valid images are added

    valid_pairs = []
    for img, mask in zip(image_filenames_train, mask_files):
        if os.path.exists(img) and os.path.exists(mask):
            valid_pairs.append((img, mask))

    df = pd.DataFrame(valid_pairs, columns=['image_filenames_train', 'mask'])
    return df

# Plot Accuracy Loss
def plot_accuracy_loss(history):
    history_post_training = history.history
    train_dice_coeff_list = history_post_training['dice_coefficients']
    test_dice_coeff_list = history_post_training['val_dice_coefficients']

    train_jaccard_list = history_post_training['iou']
    test_jaccard_list = history_post_training['val_iou']

    train_loss_list = history_post_training['loss']
    test_loss_list = history_post_training['val_loss']

    plt.figure(1)
    plt.plot(test_loss_list, 'b-')
    plt.plot(train_loss_list, 'r-')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss graph', fontsize=12)
    plt.savefig(f"results/LossGraph.png",
                    dpi=300,
                    bbox_inches='tight')
    plt.figure(2)
    plt.plot(train_dice_coeff_list, 'b-')
    plt.plot(test_dice_coeff_list, 'r-')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('accuracy graph', fontsize=12)
    plt.savefig(f"results/Accuracy_Graph.png",
                    dpi=300,
                    bbox_inches='tight')
    plt.show()

