# 🧠 Brain Tumor Segmentation using U-Net (TensorFlow/Keras)

An end-to-end deep learning pipeline for automatic brain tumor segmentation using a custom U-Net architecture trained on brain MRI images.

This project demonstrates the full medical image segmentation workflow — including model design, training, evaluation, and deployment via Hugging Face Spaces.

---

## 🚀 Live Demo

Try the interactive web application:

👉 https://huggingface.co/spaces/himanshudixit1205/brain-tumor-ai

Upload an MRI scan (.tiff / .png / .jpg) to visualize tumor segmentation results.

---

## ✨ Key Features

- Custom U-Net architecture implemented in TensorFlow/Keras  
- Hybrid Binary Cross-Entropy + Dice loss  
- Data augmentation for improved generalization  
- Modular and reproducible training pipeline  
- Deployment using Gradio on Hugging Face Spaces  

---

## 📊 Validation Metrics

| Metric | Score |
|--------|-------|
| Dice Score | ~0.77 |
| IoU | ~0.68 |

The model demonstrates stable convergence and consistent tumor region localization across validation samples.

> Note: Pixel-wise accuracy is high due to background class dominance and is not considered a primary segmentation metric.

---

## 📈 Training Curves

### Loss Curve
![Loss](results/loss_graph.png)

### Dice / IoU Curve
![Accuracy](results/accuracy_graph.png)

---

## 🖼️ Sample Predictions

![Sample 1](results/sample_0.png)  
![Sample 2](results/sample_1.png)  
![Sample 3](results/sample_2.png)

---

## 🧠 Observations

- Correct predictions on empty-mask samples (low false positives)  
- Effective detection of high-contrast tumor regions  
- Minor boundary deviations in complex tumor cases  
- Good generalization with moderate regularization  

---

## 🛠️ Tech Stack

- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Matplotlib  
- Gradio  
- Hugging Face Hub  

---

## 🤗 Model Hosting

Model weights are hosted on Hugging Face and used within the live demo application:

👉 https://huggingface.co/spaces/himanshudixit1205/brain-tumor-ai

This repository includes the complete training pipeline and inference application.

---

## 🔒 Ethics & Disclaimer

This project is intended strictly for research and educational purposes.

It is not clinically validated and must not be used for medical diagnosis or treatment decisions.

---

## 👤 Author

**Himanshu Dixit**
