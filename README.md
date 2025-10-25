# CNN Vision Explorer 🧠

An interactive Streamlit application to visualize and understand the core components of Convolutional Neural Networks (CNNs).

---

## **Creator Details**

* **Created by:** Vishav Gupta
* **Roll No.:** 102497018
* **SubGroup:** 3Q2F
* **Branch:** Computer Science & Engineering
* **Course:** UCS668 - EDGE AI: Data Center Vision

---

## **Project Modules**

This application demonstrates the CNN pipeline through four interactive modules:

1.  **🖼️ Augmentation Sandbox:**
    * Apply various image augmentations (rotation, blur, crop, etc.) in real-time.
    * See a side-by-side comparison and get the corresponding PyTorch code.

2.  **📈 Activation Function Lab:**
    * Visualize common activation functions (ReLU, Sigmoid, Tanh) and their derivatives.
    * Use the "Single Neuron Simulator" to see how inputs, weights, and biases affect neuron output and gradients.

3.  **👁️ CNN Architecture Inspector:**
    * Upload an image and select a pre-trained model (VGG16, ResNet50, MobileNetV2).
    * Visualize the feature maps from different convolutional layers to see what the network learns at various depths.

4.  **🧠 Classifier's Decision (XAI):**
    * View the model's top predictions for an image.
    * See a Grad-CAM heatmap overlay showing which parts of the image were most important for the prediction (Explainable AI).

---

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