import os
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

BASE_DEFAULT_VALUES = {
    'padding': 0,
    'rotation_angle': 0,
    'zoom_factor': 1.0,
    'flip_horizontal': False,
    'to_grayscale': False,
    'brightness_factor': 1.0,
    'blur_kernel_amount': 0,
    'blur_sigma': 1.0
}

def reset_augmentations():
    """Sets all session state keys back to their defaults, using image dimensions if available."""
    for key, value in BASE_DEFAULT_VALUES.items():
        st.session_state[key] = value

    if 'img_height' in st.session_state and 'img_width' in st.session_state:
        st.session_state['resize_height'] = st.session_state['img_height']
        st.session_state['resize_width'] = st.session_state['img_width']
        st.session_state['crop_size'] = min(st.session_state['img_height'], st.session_state['img_width'])
    else:
        st.session_state['resize_height'] = 224
        st.session_state['resize_width'] = 224
        st.session_state['crop_size'] = 224

def initialize_state_if_needed():
    """Initializes session state if keys don't exist."""
    if 'augment_state_initialized' not in st.session_state:
        reset_augmentations()
        st.session_state['augment_state_initialized'] = True


def run():
    st.title("🖼️ Module 1: The Augmentation Sandbox")
    st.write("See how data augmentation alters an image. These techniques are used to create a more robust dataset for training a model.")

    initialize_state_if_needed()

    st.header("Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image to analyze...",
        type=["jpg", "jpeg", "png"],
        key="augment_uploader"
    )

    image_loaded_this_run = False
    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            image_loaded_this_run = True

            if 'img_height' not in st.session_state or \
                st.session_state.img_height != image.height or \
                st.session_state.img_width != image.width:
                st.session_state.img_height = image.height
                st.session_state.img_width = image.width
                reset_augmentations()

                st.rerun()
        except Exception as e:
            st.error(f"Error opening image: {e}")
            uploaded_file = None
            image_loaded_this_run = False

    with st.sidebar.expander("🛠️ Augmentation Controls", expanded=True):
        st.subheader("Size & Crop")
        rh_val = st.session_state.get('resize_height', 224)
        rw_val = st.session_state.get('resize_width', 224)
        cs_val = st.session_state.get('crop_size', 224)
        pad_val = st.session_state.get('padding', 0)
        rot_val = st.session_state.get('rotation_angle', 0)
        zoom_val = st.session_state.get('zoom_factor', 1.0)
        flip_val = st.session_state.get('flip_horizontal', False)
        gray_val = st.session_state.get('to_grayscale', False)
        bright_val = st.session_state.get('brightness_factor', 1.0)
        blur_k_val = st.session_state.get('blur_kernel_amount', 0)
        blur_s_val = st.session_state.get('blur_sigma', 1.0)

        st.number_input("Resize Height", value=rh_val, min_value=1, key='resize_height')
        st.number_input("Resize Width", value=rw_val, min_value=1, key='resize_width')
        st.number_input("Center Crop Size", value=cs_val, min_value=1, key='crop_size')
        st.slider("Padding (pixels)", 0, 100, value=pad_val, step=5, key='padding')

        st.divider()
        st.subheader("Geometric Transforms")
        st.slider("Rotation Angle", -45, 45, value=rot_val, step=5, key='rotation_angle')
        st.slider("Zoom Factor", 0.5, 1.5, value=zoom_val, step=0.05, key='zoom_factor')
        st.checkbox("Horizontal Flip", value=flip_val, key='flip_horizontal')

        st.divider()
        st.subheader("Color & Filter")
        st.checkbox("Convert to Grayscale", value=gray_val, key='to_grayscale')
        st.slider("Brightness", 0.5, 1.5, value=bright_val, step=0.05, key='brightness_factor')

        st.markdown("**Gaussian Blur**")
        st.slider("Kernel Amount", 0, 25, value=blur_k_val, key='blur_kernel_amount', help="Calculates an odd kernel size. 0 means no blur.")
        st.slider("Sigma (Intensity)", 0.1, 20.0, value=blur_s_val, step=0.1, key='blur_sigma')

        st.divider()
        st.button("Reset Controls", on_click=reset_augmentations, key="reset_button")

    if image_loaded_this_run and image is not None:
        st.success("Image uploaded successfully!")

        image_tensor = F.to_tensor(image)
        transforms_to_apply = []

        if st.session_state.resize_height != st.session_state.img_height or st.session_state.resize_width != st.session_state.img_width:
            transforms_to_apply.append(T.Resize((st.session_state.resize_height, st.session_state.resize_width)))
            current_h, current_w = st.session_state.resize_height, st.session_state.resize_width
        else:
            current_h, current_w = st.session_state.img_height, st.session_state.img_width

        if st.session_state.crop_size < min(current_h, current_w):
            transforms_to_apply.append(T.CenterCrop(st.session_state.crop_size))
        if st.session_state.rotation_angle != 0:
            transforms_to_apply.append(T.RandomRotation(degrees=(st.session_state.rotation_angle, st.session_state.rotation_angle)))
        if st.session_state.zoom_factor != 1.0:
            scale_value = 1.0 / st.session_state.zoom_factor if st.session_state.zoom_factor != 0 else 1.0
            transforms_to_apply.append(T.RandomAffine(degrees=0, scale=(scale_value, scale_value)))
        if st.session_state.flip_horizontal:
            transforms_to_apply.append(T.RandomHorizontalFlip(p=1.0))
        if st.session_state.to_grayscale:
            transforms_to_apply.append(T.Grayscale(num_output_channels=3))
        if st.session_state.brightness_factor != 1.0:
            transforms_to_apply.append(T.ColorJitter(brightness=st.session_state.brightness_factor))
        if st.session_state.blur_kernel_amount > 0:
            kernel_size = st.session_state.blur_kernel_amount * 2 + 1
            transforms_to_apply.append(T.GaussianBlur(kernel_size=kernel_size, sigma=st.session_state.blur_sigma))
        if st.session_state.padding > 0:
            transforms_to_apply.append(T.Pad(padding=st.session_state.padding))

        if transforms_to_apply:
            augmentation_pipeline = T.Compose(transforms_to_apply)
            augmented_tensor = augmentation_pipeline(image_tensor)
        else:
            augmented_tensor = image_tensor

        augmented_image_array = augmented_tensor.permute(1, 2, 0).numpy()
        augmented_image_array = np.clip(augmented_image_array * 255, 0, 255).astype(np.uint8)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            st.subheader("Augmented Image")
            st.image(augmented_image_array, caption="Augmented", use_column_width=True)

        with st.expander("Show Generated Code"):
            code_lines = ["import torchvision.transforms as T", "\naugmentation_pipeline = T.Compose(["]
            if st.session_state.resize_height != st.session_state.img_height or st.session_state.resize_width != st.session_state.img_width: code_lines.append(f"    T.Resize(({st.session_state.resize_height}, {st.session_state.resize_width})),")
            if st.session_state.crop_size < min(current_h, current_w): code_lines.append(f"    T.CenterCrop({st.session_state.crop_size}),")
            if st.session_state.rotation_angle != 0: code_lines.append(f"    T.RandomRotation(degrees=({st.session_state.rotation_angle}, {st.session_state.rotation_angle})),")
            if st.session_state.zoom_factor != 1.0:
                scale_value = 1.0 / st.session_state.zoom_factor if st.session_state.zoom_factor != 0 else 1.0
                code_lines.append(f"    T.RandomAffine(degrees=0, scale=({scale_value:.2f}, {scale_value:.2f})),")
            if st.session_state.flip_horizontal: code_lines.append("    T.RandomHorizontalFlip(p=1.0),")
            if st.session_state.to_grayscale: code_lines.append("    T.Grayscale(num_output_channels=3),")
            if st.session_state.brightness_factor != 1.0: code_lines.append(f"    T.ColorJitter(brightness={st.session_state.brightness_factor:.2f}),")
            if st.session_state.blur_kernel_amount > 0:
                kernel_size = st.session_state.blur_kernel_amount * 2 + 1
                code_lines.append(f"    T.GaussianBlur(kernel_size={kernel_size}, sigma={st.session_state.blur_sigma:.1f}),")
            if st.session_state.padding > 0: code_lines.append(f"    T.Pad(padding={st.session_state.padding}),")
            code_lines.append("])")
            st.code('\n'.join(code_lines), language='python')

    elif not image_loaded_this_run :
        st.info("Please upload an image file to begin.")