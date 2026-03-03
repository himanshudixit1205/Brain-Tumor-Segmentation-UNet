# Environment Configuration
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Imports
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

tf.config.set_visible_devices([], "GPU")

# Configuration
MODEL_REPO = os.getenv("HF_MODEL_REPO")
IMAGE_SIZE = 128
THRESHOLD = 0.3

# Load Model
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_dice.keras"
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# Preprocessing
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# Display Normalization
def normalize_for_display(img_np):
    img_np = img_np.astype(np.float32)

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    return (img_np * 255).astype(np.uint8)

# Overlay Creation
def create_overlay(image, mask):
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 0] = mask
    return np.clip(overlay * 0.7 + red * 0.6, 0, 255).astype(np.uint8)

# Prediction
def predict(image):
    if image is None:
        return None, None, "No image uploaded"

    img_np = np.array(image)
    preview = normalize_for_display(img_np)

    input_img = preprocess(Image.fromarray(preview))
    prediction = model.predict(input_img, verbose=0)[0]

    mask = (prediction > THRESHOLD).astype(np.uint8)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = (mask * 255).astype(np.uint8)

    mask_resized = Image.fromarray(mask).resize(
        (preview.shape[1], preview.shape[0]),
        resample=Image.NEAREST
    )
    mask_resized = np.array(mask_resized)

    overlay = create_overlay(preview, mask_resized)

    diagnosis = "⚠️ Tumor Detected" if np.sum(mask) > 0 else "✅ No Tumor Detected"

    return preview, overlay, diagnosis

# User Interface
title = "🧠 AI Brain Tumor Segmentation Demo"

description = """
Deep learning–based UNet segmentation model for brain MRI analysis.

Validation Performance:
- Dice Score ≈ 0.75
- Accuracy ≈ 75%

Educational and research use only.
Not intended for clinical diagnosis.
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    input_image = gr.Image(type="pil", label="Upload MRI Scan")
    run_btn = gr.Button("Run Analysis", variant="primary")

    diagnosis_box = gr.Textbox(label="AI Diagnosis", lines=2)

    with gr.Row():
        original_output = gr.Image(label="Original MRI")
        overlay_output = gr.Image(label="Tumor Highlight Overlay")

    run_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[original_output, overlay_output, diagnosis_box],
    )

# Launch
demo.launch(server_name="0.0.0.0", server_port=7860)