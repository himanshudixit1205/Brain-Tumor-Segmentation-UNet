## 🧠 Brain Tumor Segmentation using U-Net (TensorFlow/Keras)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Model](https://img.shields.io/badge/Model-U--Net-blueviolet)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-red)
![Research](https://img.shields.io/badge/Research-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?logo=huggingface)](https://huggingface.co/spaces/himanshudixit1205/brain-tumor-ai)

An end-to-end deep learning pipeline for automated brain tumor segmentation from MRI scans using a custom U-Net architecture.

This project covers the full medical imaging workflow — including data preprocessing, model training, evaluation using segmentation-specific metrics (Dice & IoU), 
and deployment via an interactive web application.

------------------------------------------------------------------------

## 🚀 Live Demo

Interactive web application:

👉 https://huggingface.co/spaces/himanshudixit1205/brain-tumor-ai

Upload an MRI scan (.tiff / .png / .jpg) to visualize tumor segmentation
results.

------------------------------------------------------------------------

## ✨ Key Features

-   Custom U-Net architecture implemented in TensorFlow/Keras\
-   Hybrid Binary Cross-Entropy + Dice loss\
-   Data augmentation for improved generalization\
-   Modular and reproducible training pipeline\
-   Gradio-based inference app deployed on Hugging Face Spaces

------------------------------------------------------------------------

## 📦 Dataset

The model was trained on the **Kaggle Brain MRI Segmentation dataset**
containing **~3900 MRI image–mask pairs with corresponding tumor masks**.

Preprocessing steps:
- Image resizing to 128×128 resolution
- Pixel normalization
- Data augmentation (rotation, shifts, zoom, flips)
- Train / validation / test split

------------------------------------------------------------------------

## 📊 Validation Metrics

  Metric       Score
  ------------ --------
  Dice Score   \~0.77
  IoU          \~0.68

The model demonstrates stable convergence and consistent tumor
localization across validation samples.

> Note: Pixel-wise accuracy is typically high in segmentation tasks due
> to background dominance and is not considered a primary evaluation
> metric.

------------------------------------------------------------------------

## 📈 Training Curves

### Loss Curve

![Loss](results/loss_graph.png)

### Dice / IoU Curve

![Accuracy](results/accuracy_graph.png)

------------------------------------------------------------------------

## 🖼️ Sample Predictions

![Sample 1](results/sample_0.png)\
![Sample 2](results/sample_1.png)\
![Sample 3](results/sample_2.png)

------------------------------------------------------------------------

## 📜 Project Structure

    .
    ├── src/
    │   ├── train.py        # Training pipeline
    │   ├── unet.py         # Model architecture
    │   ├── utils.py        # Data loading & metrics
    │
    ├── app.py              # Gradio inference app
    ├── config.yaml         # Training configuration
    ├── results/            # Training plots & samples
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   TensorFlow / Keras
-   NumPy
-   OpenCV
-   Matplotlib
-   Scikit-learn
-   Gradio
-   Hugging Face Hub

------------------------------------------------------------------------

## 🤗 Model Hosting

Model weights are hosted on Hugging Face and used within the live demo
application:

👉 https://huggingface.co/spaces/himanshudixit1205/brain-tumor-ai

This repository includes the complete training pipeline and inference
application.

------------------------------------------------------------------------

## 🔒 Ethics & Disclaimer

This project is intended strictly for research and educational purposes.

It is not clinically validated and must not be used for medical
diagnosis or treatment decisions.

Always consult a qualified medical professional for clinical evaluation.

------------------------------------------------------------------------

## 👤 Author

**Himanshu Dixit**

------------------------------------------------------------------------

## ⭐ If You Found This Useful

Feel free to star the repository or connect with me.
