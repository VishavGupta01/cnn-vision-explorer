import os
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def load_model(model_name):
    with st.spinner(f"Loading {model_name}... This may take a moment."):
        if model_name == "VGG16":
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        elif model_name == "ResNet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == "MobileNetV2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.eval()
        return model

def get_conv_layers(model):
    """Finds all nn.Conv2d layers and returns their names."""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(name)
    return conv_layers

def preprocess_image(image):
    """Prepares the image for the model."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

g_feature_map = {}
def get_features_hook(module, input, output):
    """A 'hook' function that PyTorch will call during the forward pass."""
    g_feature_map['output'] = output.detach()

def plot_feature_maps(features):
    """Plots the top 12 most activated feature maps."""
    features = features.squeeze(0)
    num_channels = features.shape[0]
    num_to_show = min(num_channels, 12)

    mean_activations = features.mean(dim=[1, 2])
    top_indices = mean_activations.cpu().argsort(descending=True)[:num_to_show]

    rows = (num_to_show + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(top_indices):
        ax = axes[i]
        feature_map = features[idx].cpu().numpy()
        im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f"Channel {idx}")
        ax.axis('off')

    for i in range(num_to_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)

MODEL_DESCRIPTIONS = {
    "VGG16": """
    **VGG16** is a classic and straightforward architecture...
    """,
    "ResNet50": """
    **ResNet50** (Residual Network) revolutionized deep learning...
    """,
    "MobileNetV2": """
    **MobileNetV2** is designed for efficiency (e.g., on mobile phones)...
    """
}

def run():
    st.title("👁️ Module 3: The CNN Architecture Inspector")
    st.write("Upload an image, choose a model, and see what it 'sees' inside its hidden layers.")

    st.header("1. Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="inspector_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.header("2. Select a Model and Layer")

        model_name = st.selectbox(
            "Choose a Model:",
            ["VGG16", "ResNet50", "MobileNetV2"],
            key="inspector_model_select"
        )
        model = load_model(model_name)

        with st.expander("About this model"):
            st.markdown(MODEL_DESCRIPTIONS[model_name])

        conv_layers = get_conv_layers(model)
        layer_name = st.selectbox(
            "Choose a Convolutional Layer to Visualize:",
            conv_layers,
            key="inspector_layer_select"
        )

        with st.expander("Simplified Model Architecture"):
            st.info(f"You selected: {layer_name}")
            st.code(str(model))

        st.header("3. Visualize")
        if st.button("Run Model and Visualize Features", key="inspector_run_button"):
            with st.spinner("Processing image and extracting features..."):
                processed_image = preprocess_image(image)

                hook = model.get_submodule(layer_name).register_forward_hook(get_features_hook)

                with torch.no_grad():
                    model(processed_image)

                hook.remove()

                st.subheader(f"Top {min(g_feature_map['output'].shape[1], 12)} Most Activated Feature Maps from {layer_name}")
                plot_feature_maps(g_feature_map['output'])

                st.subheader("What am I looking at?")
                explanation = ""

                if model_name == "VGG16":
                    if layer_name in ["features.0", "features.2"]:
                         explanation = "### 📍 VGG16: Block 1 (Early Layer)\n\nFilters are learning basic **colors, simple edges, and gradients.**"
                    elif layer_name in ["features.5", "features.7"]:
                         explanation = "### 📍 VGG16: Block 2\n\nFeatures combine into **basic corners, curves, and simple textures.**"
                    elif layer_name in ["features.10", "features.12", "features.14"]:
                         explanation = "### 📍 VGG16: Block 3 (Mid-Layer)\n\nFilters respond to complex patterns like **mesh, dots, or simple object parts.**"
                    elif layer_name in ["features.17", "features.19", "features.21"]:
                         explanation = "### 📍 VGG16: Block 4\n\nFeatures respond to recognizable parts like **curves of wheels or shapes of eyes.**"
                    else:
                         explanation = "### 📍 VGG16: Block 5 (Deep Layer)\n\nHighly specialized features detect complex concepts like **'dog snout' or 'building facade'.**"

                elif model_name == "ResNet50":
                    if layer_name == "conv1":
                        explanation = "### 📍 ResNet50: Initial Convolution (Early Layer)\n\nLarge filter captures basic **blobs of color and strong edges.**"
                    elif layer_name.startswith("layer1"):
                        explanation = "### 📍 ResNet50: Layer 1 (Residual Block)\n\nCombines initial features. **Skip connections** preserve original info."
                    elif layer_name.startswith("layer2") or layer_name.startswith("layer3"):
                        explanation = "### 📍 ResNet50: Layer 2 & 3 (Mid-Layers)\n\nBuilding rich features: **textures, patterns, complex object parts.**"
                    else:
                        explanation = "### 📍 ResNet50: Layer 4 (Deep Layer)\n\nAbstract, class-specific features like **'feathery texture' or 'headlight shape'.**"

                elif model_name == "MobileNetV2":
                    layer_index = conv_layers.index(layer_name)
                    if layer_index < 2:
                        explanation = "### 📍 MobileNetV2: Initial Convolutions (Early Layer)\n\nLightweight stem captures basic **edges and colors.**"
                    elif layer_index < 12:
                        explanation = "### 📍 MobileNetV2: Inverted Residuals (Mid-Layers)\n\nUses efficient **'depthwise-separable' convolutions** to learn patterns."
                    else:
                        explanation = "### 📍 MobileNetV2: Inverted Residuals (Deep Layers)\n\nHighly abstract features optimized for efficient classification, using **a fraction of the parameters.**"

                st.info(explanation)

    else:
        st.info("Please upload an image to begin.")