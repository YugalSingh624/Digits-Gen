# Filename: app.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, one_hot

device = torch.device("cpu")

# Load the trained generator
generator = Generator().to(device)
generator.load_state_dict(torch.load("cgan_digit_generator_kaggle.pth", map_location=device))
generator.eval()

# Streamlit UI
st.title("🧠 Handwritten Digit Generator")
st.markdown("Generate MNIST-style images using a **Conditional GAN** model.")

digit = st.selectbox("Select a digit (0–9):", list(range(10)))
num_images = st.slider("Number of images to generate:", min_value=1, max_value=20, value=5)

if st.button("Generate"):
    noise = torch.randn(num_images, 100)
    labels = torch.tensor([digit] * num_images)
    onehot_labels = one_hot(labels)

    with torch.no_grad():
        generated_imgs = generator(noise, onehot_labels).cpu().squeeze().numpy()

    st.subheader(f"Generated Digit: {digit}")
    cols = st.columns(min(num_images, 5))
    for i in range(num_images):
        img = generated_imgs[i]
        if img.ndim == 2:
            cols[i % 5].image(img, width=100, clamp=True, channels='L')
        else:
            cols[i % 5].image(img[0], width=100, clamp=True, channels='L')
