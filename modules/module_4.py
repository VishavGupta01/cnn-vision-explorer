import os
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F
import cv2

@st.cache_resource
def load_model(model_name):
    with st.spinner(f"Loading {model_name}..."):
        weights = None
        model = None
        if model_name == "VGG16":
            weights = models.VGG16_Weights.DEFAULT
            model = models.vgg16(weights=weights)
        elif model_name == "ResNet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        elif model_name == "MobileNetV2":
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)

        model.eval()
        return model, weights

def preprocess_image(image):
    """Prepares the image for the model."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def set_inplace_false(model):
    """
    Recursively loops through all modules in a model and sets
    any 'inplace' attribute to False.
    """
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_features)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.features = output.detach().clone()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach().clone()

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        score = output[0, target_class]
        score.backward()

        grads = self.gradients
        features = self.features

        weights = F.adaptive_avg_pool2d(grads, 1)
        heatmap = (features * weights).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap.squeeze().cpu().numpy()

def overlay_heatmap(image, heatmap):
    """Overlays a CAM heatmap onto the original image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_np = np.array(image)
    heatmap_np = (heatmap * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_color, (img_np.shape[1], img_np.shape[0]))

    overlayed_img = cv2.addWeighted(img_np, 0.5, heatmap_resized, 0.5, 0)

    return overlayed_img

def run():
    st.title("🧠 Module 4: The Classifier's Decision (XAI)")
    st.write("This is the finale. We run the model, see its prediction, and use Grad-CAM to see *why* it made that choice.")

    st.header("1. Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="xai_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        st.header("2. Select a Model")

        model_name = st.selectbox(
            "Choose a Model:",
            ["VGG16", "ResNet50", "MobileNetV2"],
            key="xai_model_select"
        )
        model, weights = load_model(model_name)
        class_names = weights.meta["categories"]

        if model_name == "VGG16":
            target_layer = model.features[28]
        elif model_name == "ResNet50":
            target_layer = model.layer4[-1]
        elif model_name == "MobileNetV2":
            target_layer = model.features[-1]

        st.header("3. Run Analysis")

        if st.button("Find out what the model is thinking...", key="xai_run_button"):
            with st.spinner("Analyzing image..."):

                input_tensor = preprocess_image(image)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)

                top5_probs, top5_indices = torch.topk(probabilities, 5)
                top5_probs = top5_probs.squeeze().tolist()
                top5_indices = top5_indices.squeeze().tolist()
                top5_labels = [class_names[idx] for idx in top5_indices]

                set_inplace_false(model)

                grad_cam = GradCAM(model, target_layer)
                heatmap = grad_cam.generate_heatmap(input_tensor, target_class=top5_indices[0])

                overlayed_image = overlay_heatmap(image, heatmap)

                st.subheader(f"The model's top prediction is: **{top5_labels[0]}**")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(overlayed_image, caption="What the model 'sees' (Grad-CAM)", use_container_width=True)

                st.info(f"""
                **What is this?** The heatmap on the right shows *why* the model
                predicted **'{top5_labels[0]}'**.
                The bright red areas are the pixels the model focused on the most.
                """)

                st.subheader("Top 5 Predictions")

                fig = go.Figure(go.Bar(
                    x=[p * 100 for p in top5_probs],
                    y=top5_labels,
                    orientation='h',
                    text=[f"{p*100:.1f}%" for p in top5_probs],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Model Confidence",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Prediction",
                    yaxis=dict(autorange="reversed"),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please upload an image to begin.")