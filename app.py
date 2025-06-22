# Filename: app.py
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Generator, one_hot

# Load the trained generator
generator = Generator()
generator.load_state_dict(torch.load("digit_generator.pth", map_location=torch.device('cpu')))
generator.eval()

# Web UI
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")
digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate Images"):
    noise = torch.randn(5, 64)
    onehot = one_hot(torch.tensor([digit]*5))
    with torch.no_grad():
        generated_imgs = generator(noise, onehot).squeeze().numpy()

    st.write(f"Generated images of digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        cols[i].image(generated_imgs[i], width=100, clamp=True, channels='L')
