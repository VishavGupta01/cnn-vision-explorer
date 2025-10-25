import streamlit as st
from streamlit_option_menu import option_menu

import modules.module_1
import modules.module_2
import modules.module_3
import modules.module_4

def load_css(file_name):
    """Loads a CSS file from the assets folder."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def display_footer():
    """Displays a consistent footer."""
    st.divider()
    st.markdown(
        """
        ---
        **Created by:** Vishav Gupta   |   **Roll No.:** 102497018   |   **SubGroup:** 3Q2F   |   **Branch:** Computer Science & Engineering   |   **Course:** UCS668 - EDGE AI: Data Center Vision   |   **Project:** CNN Vision Explorer
        """
        , unsafe_allow_html=True
    )

st.set_page_config(
    page_title="CNN Vision Explorer",
    page_icon="🧠",
    layout="wide"
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Augmentation", "Activations", "Inspector", "XAI"],
    icons=["house", "image-alt", "graph-up", "search", "lightbulb"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "#F8F9FA",
            "border-bottom": "1px solid #E9ECEF"
        },
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin":"0px",
            "color": "#9B59B6",
            "background-color": "#E9ECEF",
            "--hover-color": "#D3D3D3",
            "icon-color": "#9B59B6",
        },
        "nav-link-selected": {
            "background-color": "#9B59B6",
            "color": "#FFFFFF",
            "font-weight": "bold",
            "icon-color": "#FFFFFF",
        },
    }
)

with st.container():

    if selected == "Home":
        st.title("Welcome to the CNN Vision Explorer 🚀")
        st.markdown("This interactive web app helps you understand Convolutional Neural Networks (CNNs).")
        st.markdown("Select a module from the navbar above to begin.")
        st.divider()
        st.header("Project Modules")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("🖼️ 1. Augmentation Sandbox")
                st.write("Interactively apply image augmentations (rotation, blur, etc.) and see the PyTorch code.")

            with st.container(border=True):
                st.subheader("👁️ 3. CNN Architecture Inspector")
                st.write("Visualize feature maps from different layers of pre-trained models like VGG16 or ResNet50.")

        with col2:
            with st.container(border=True):
                st.subheader("📈 2. Activation Function Lab")
                st.write("Explore activation functions (ReLU, Sigmoid) and their gradients with an interactive neuron simulator.")

            with st.container(border=True):
                st.subheader("🧠 4. Classifier's Decision (XAI)")
                st.write("See *why* a model makes its prediction using Grad-CAM heatmaps for explainable AI.")


    elif selected == "Augmentation":
        modules.module_1.run()

    elif selected == "Activations":
        modules.module_2.run()

    elif selected == "Inspector":
        modules.module_3.run()

    elif selected == "XAI":
        modules.module_4.run()

display_footer()