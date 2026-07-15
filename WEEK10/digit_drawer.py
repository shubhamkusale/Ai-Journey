import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# Same CNN architecture as cnn_mnist.py — must match exactly
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model once, cache it so it doesn't reload every interaction
@st.cache_resource
def load_model():
    model = CNN()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, 'mnist_cnn.pth'),
        map_location='cpu'
    ))
    model.eval()
    return model

model = load_model()

# App UI
st.title("🔢 Digit Drawer")
st.write("Draw any digit (0–9) below — CNN predicts it instantly!")

# Drawing canvas — white pen on black background (like MNIST)
canvas_result = st_canvas(
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# When something is drawn
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Only predict if canvas isn't empty
    if img[:, :, :3].sum() > 0:

        # Convert to grayscale PIL image
        img_pil = Image.fromarray(img[:, :, :3].astype('uint8')).convert('L')

        # Resize to 28x28 (MNIST size)
        img_resized = img_pil.resize((28, 28))

        # Normalize same way as training
        img_array = np.array(img_resized) / 255.0
        img_normalized = (img_array - 0.5) / 0.5

        # Convert to tensor shape (1, 1, 28, 28)
        img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)

        # Get probabilities
        probs = torch.nn.functional.softmax(output[0], dim=0)
        predicted = torch.argmax(probs).item()
        confidence = probs[predicted].item() * 100

        # Show result
        st.markdown(f"## Predicted: **{predicted}**")
        st.markdown(f"Confidence: **{confidence:.1f}%**")

        st.write("All digit confidences:")
        for i, p in enumerate(probs):
            st.progress(float(p), text=f"Digit {i}: {p*100:.1f}%")