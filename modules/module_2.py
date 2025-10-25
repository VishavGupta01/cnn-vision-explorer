import streamlit as st
import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

ACTIVATION_FUNCTIONS = {
    "ReLU": {
        "func": relu,
        "derivative": relu_derivative,
        "explanation": """
        **ReLU (Rectified Linear Unit)** is the most popular activation function.
        - **Pros:** Simple, fast, avoids vanishing gradients for positive inputs.
        - **Cons:** Suffers from the "Dying ReLU" problem (zero gradient for negative inputs).
        """
    },
    "Leaky ReLU": {
        "func": leaky_relu,
        "derivative": leaky_relu_derivative,
        "explanation": """
        **Leaky ReLU** attempts to fix the "Dying ReLU" problem.
        - **Pros:** Allows a small, non-zero gradient for negative inputs, preventing neurons from dying.
        - **Cons:** Slightly more complex than ReLU.
        """
    },
    "Sigmoid": {
        "func": sigmoid,
        "derivative": sigmoid_derivative,
        "explanation": """
        **Sigmoid** is rarely used in hidden layers now.
        - **Pros:** Outputs between 0 and 1, useful for binary classification outputs.
        - **Cons:** Suffers badly from **Vanishing Gradients** at the extremes.
        """
    },
    "Tanh": {
        "func": tanh,
        "derivative": tanh_derivative,
        "explanation": """
        **Tanh** is similar to Sigmoid but zero-centered.
        - **Pros:** Outputs between -1 and 1, zero-centered output can help learning.
        - **Cons:** Also suffers from **Vanishing Gradients** at the extremes.
        """
    }
}

def run():
    st.title("📈 Module 2: The Activation Function Lab")
    st.write("Visualize activation functions and their derivatives to understand their impact on learning.")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_function_name = st.selectbox(
            "Choose an activation function:",
            list(ACTIVATION_FUNCTIONS.keys()),
            key="activation_func_select"
        )
    with col2:
        st.write("")
        st.write("")
        show_derivative = st.checkbox("Show Derivative", True, key="activation_show_deriv")

    selected_func_dict = ACTIVATION_FUNCTIONS[selected_function_name]
    func = selected_func_dict["func"]
    derivative = selected_func_dict["derivative"]
    explanation = selected_func_dict["explanation"]

    x_range = np.linspace(-5, 5, 400)
    y_range = func(x_range)
    y_prime_range = derivative(x_range)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name=f"f(x) = {selected_function_name}(x)",
        line=dict(color='blue', width=3),
        hovertemplate="x: %{x:.2f}<br>f(x): %{y:.2f}"
    ))

    if show_derivative:
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_prime_range,
            mode='lines',
            name=f"f'(x) = Derivative",
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate="x: %{x:.2f}<br>f'(x): %{y:.2f}"
        ))

    fig.update_layout(
        title=f"{selected_function_name} Function and its Derivative",
        xaxis_title="Input Value (x)",
        yaxis_title="Output Value (f(x) or f'(x))",
        legend_title="Function",
        hovermode="x unified",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[-2, 2])
    )

    with st.expander("Learn about this function"):
        st.markdown(explanation)
        st.markdown("""
        ---
        - The **Blue Line (f(x))** is the activation function.
        - The **Red Line (f'(x))** is the derivative (or gradient). Its value tells the network how to update its weights.
            - **High Derivative:** Fast learning 🚀
            - **Low Derivative:** Slow learning 🐌
            - **Zero Derivative:** No learning (Stuck!) ⛔
        """)

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")
    st.header("⚡ Single Neuron Simulator")
    st.write("See how a single neuron responds using the selected activation function.")

    col1, col2, col3 = st.columns(3)
    with col1:
        x_input = st.slider("Input (x)", -3.0, 3.0, 1.0, 0.1, key="neuron_x")
    with col2:
        weight = st.slider("Weight (w)", -2.0, 2.0, 0.5, 0.1, key="neuron_w")
    with col3:
        bias = st.slider("Bias (b)", -2.0, 2.0, 0.0, 0.1, key="neuron_b")

    z = (weight * x_input) + bias
    activation = func(z)
    gradient = derivative(z)

    st.subheader("Neuron's Internal Calculation:")
    st.markdown(f"**Linear Step (z):** `({weight:.1f} * {x_input:.1f}) + {bias:.1f}` = **`{z:.2f}`**")
    st.markdown(f"**Activation Step (a):** `{selected_function_name}({z:.2f})` = **`{activation:.2f}`**")

    st.subheader("How this impacts learning:")
    st.metric(label="Gradient at this point (f'(z))", value=f"{gradient:.2f}")

    if gradient == 0:
        st.warning("⛔ **Learning is STUCK!** Gradient is zero.")
    elif abs(gradient) < 0.1:
        st.info("🐌 **Learning is slow.** Gradient is very small.")
    else:
        st.success("🚀 **Learning is fast!** Gradient is steep.")

    fig.add_vline(x=z, line_width=3, line_dash="solid", line_color="green", annotation_text="Neuron (z)", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=[z],
        y=[activation],
        mode='markers',
        marker=dict(color='green', size=15, symbol='star'),
        name="Neuron's Output (a)",
        hovertemplate=f"z: {z:.2f}<br>a: {activation:.2f}"
    ))

    st.plotly_chart(fig, use_container_width=True)