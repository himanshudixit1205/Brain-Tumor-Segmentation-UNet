# 🧠 Brain Tumor Segmentation using U-Net (TensorFlow/Keras)

A deep learning project for automatic brain tumor segmentation using a custom U-Net architecture trained on brain MRI scans.

This project demonstrates an end-to-end medical image segmentation pipeline — from model training to deployment via Hugging Face Spaces.

---

## 🚀 Live Demo

Try the interactive web app:

👉 https://huggingface.co/spaces/himanshudixit1205/brain-tumor-ai

Upload an MRI scan (.tiff / .png / .jpg) to visualize tumor segmentation results.

---

## ✨ Highlights

- Custom U-Net implementation (TensorFlow/Keras)
- Hybrid BCE + Dice loss
- Dice Score ~0.77–0.78
- Strong tumor localization
- Clean modular code structure
- Deployed using Hugging Face Spaces

---

## 📊 Final Metrics

| Metric | Score |
|--------|-------|
| Dice Score | ~0.77 |
| IoU | ~0.68 |
| Validation Accuracy | ~99.6% |

The model shows stable convergence and reliable tumor region segmentation across validation samples.

---

## 📈 Training Curves

### Loss Curve
![Loss](results/loss_graph.png)

### Accuracy Curve
![Accuracy](results/accuracy_graph.png)

---

## 🖼️ Sample Predictions

![Sample 1](results/sample_0.png)  
![Sample 2](results/sample_1.png)  
![Sample 3](results/sample_2.png)

---

## 🧠 Observations

- Correct predictions on empty-mask images (low false positives)
- Strong tumor detection in high-contrast regions
- Minor boundary shifts in complex tumor cases
- Good generalization without heavy regularization

---

## 🛠️ Tech Stack

- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Matplotlib  
- Gradio  
- Hugging Face Hub  

---

## 🤗 Model Weights

Model weights are hosted privately on Hugging Face.

Live demo available here:  
👉 https://huggingface.co/spaces/himanshudixit1205/brain-tumor-ai

---

## 🔒 Ethics & Safety

⚠️ For research & educational use only — not a medical diagnosis.

Due to medical AI considerations, model weights are not publicly released.  
This repository provides full training code and a live inference demo.

---

## 👤 Author

**Himanshu Dixit**  
