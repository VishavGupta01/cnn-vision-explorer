# CNN Vision Explorer 🧠

An interactive Streamlit application to visualize and understand the core components of Convolutional Neural Networks (CNNs).

---

## 🌐 Live Demo

**Access the deployed application here:** [https://cnn-vision-explorer.streamlit.app](https://cnn-vision-explorer.streamlit.app)

---

## **Creator Details**

* **Created by:** Vishav Gupta
* **Roll No.:** 102497018
* **SubGroup:** 3Q2F
* **Branch:** Computer Science & Engineering
* **Course:** UCS668 - EDGE AI: Data Center Vision

---

## ✨ Features

This application is divided into four distinct modules:

1.  **🖼️ Augmentation Sandbox:**
    * Interactively apply various image augmentation techniques (rotation, brightness, blur, crop, etc.) to an uploaded image.
    * See a real-time side-by-side comparison of the original and augmented images.
    * Generates the corresponding PyTorch code snippet for the selected transformations.

2.  **📈 Activation Function Lab:**
    * Visualize common activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU) and their derivatives.
    * Use the interactive "Single Neuron Simulator" to see how input, weight, and bias affect a neuron's output and gradient, demonstrating concepts like vanishing gradients and the dying ReLU problem.

3.  **👁️ CNN Architecture Inspector:**
    * Upload an image and select a pre-trained CNN model (VGG16, ResNet50, MobileNetV2).
    * Explore the model's architecture and choose specific convolutional layers.
    * Visualize the top activated **feature maps** from the selected layer to understand hierarchical feature extraction (edges -> textures -> parts -> objects).
    * Includes model-specific explanations of what features are typically learned at different depths.

4.  **🧠 Classifier's Decision (XAI):**
    * See the model's top 5 predictions for your uploaded image, displayed with confidence scores in a bar chart.
    * Implements **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate a heatmap overlay.
    * This heatmap visually explains *which parts* of the image the model focused on to make its top prediction, providing **Explainable AI (XAI)**.

---

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Deep Learning:** PyTorch, Torchvision
* **Plotting:** Plotly, Matplotlib
* **Image Processing:** OpenCV, Pillow
* **Numerical:** NumPy
* **UI/Navigation:** streamlit-option-menu

---

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VishavGupta01/cnn-vision-explorer
    cd cnn-vision-explorer
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Running the App

1.  Navigate to the project directory in your terminal.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  The application should automatically open in your web browser.

---

## **Project Structure**
```
CNN_Vision_Explorer/
│
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
│
├── assets/
│   └── style.css           # Custom CSS
│
├── modules/
│   ├── module_1.py         # Code for Augmentation Sandbox
│   ├── module_2.py         # Code for Activation Function Lab
│   ├── module_3.py         # Code for CNN Architecture Inspector
│   └── module_4.py         # Code for Classifier's Decision (XAI)
│
├── app.py                  # Main application script with navbar & routing
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---