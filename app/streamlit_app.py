
# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import time
import io

from app.simulator import LinearRegressionGD
from app.utils import generate_linear_data

import matplotlib.pyplot as plt

st.set_page_config(page_title="Linear Regression Simulator", layout="wide")

st.title("Linear Regression Simulator — Real-time (Streamlit)")

# Sidebar: controls
with st.sidebar:
    st.header("Data / True function")
    true_m = st.slider("True slope (m)", min_value=-10.0, max_value=10.0, value=4.0, step=0.1)
    true_b = st.slider("True intercept (b)", min_value=-50.0, max_value=50.0, value=7.0, step=0.5)
    sample_size = st.slider("Sample size", min_value=5, max_value=500, value=80, step=1)
    noise = st.slider("Noise (std)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    seed = st.number_input("Random seed (0 = deterministic)", min_value=0, step=1, value=0)

    st.markdown("---")
    st.header("Training")
    lr = st.number_input("Learning rate", min_value=1e-6, max_value=10.0, value=0.01, format="%.6f")
    epochs = st.slider("Epochs", min_value=1, max_value=5000, value=200, step=1)
    update_every = st.slider("Update plot every N epochs (lower = more realtime)", min_value=1, max_value=max(1, epochs//1), value=max(1, epochs//50))
    speed = st.slider("Animation delay (seconds)", min_value=0.0, max_value=1.0, value=0.01, step=0.01)

    st.markdown("---")
    st.header("Options")
    show_loss = st.checkbox("Show loss curve", value=True)
    compare_lrs = st.checkbox("Compare multiple learning rates (run separate)", value=False)
    if compare_lrs:
        lr_list_text = st.text_input("Comma-separated LR list (e.g. 0.0005,0.001,0.01,0.1)", value="0.0005,0.001,0.01,0.1")

    st.markdown("---")
    st.markdown("**Data upload**")
    uploaded = st.file_uploader("Upload CSV with columns `x` and `y` (optional)", type=["csv"])
    st.markdown("Or click `Randomize data` to generate synthetic data.")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    plot_placeholder = st.empty()
    loss_placeholder = st.empty()

with col2:
    st.subheader("Controls")
    run_button = st.button("Train (real-time)")
    randomize_button = st.button("Randomize data")
    stop_button = st.button("Stop training")
    st.write("Model parameters (live):")
    param_text = st.empty()
    st.write("Final stats:")
    final_text = st.empty()
    st.write("Export")
    csv_btn = st.button("Download trained predictions (CSV)")

# Prepare data
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        X = df["x"].to_numpy()
        y = df["y"].to_numpy()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        X, y = generate_linear_data(true_m, true_b, sample_size, noise, seed)
else:
    X, y = generate_linear_data(true_m, true_b, sample_size, noise, seed if seed != 0 else 42)

# Allow on-page quick edits through a small dataframe
data_df = pd.DataFrame({"x": X, "y": y})
edited = st.experimental_data_editor(data_df, num_rows="dynamic")
if edited is not None:
    try:
        X = edited["x"].to_numpy(dtype=float)
        y = edited["y"].to_numpy(dtype=float)
    except Exception:
        pass

# Compare multiple LRs option handler
if compare_lrs and st.button("Run LR comparison"):
    try:
        lr_list = [float(x.strip()) for x in lr_list_text.split(",") if x.strip() != ""]
    except:
        st.error("Invalid LR list")
        lr_list = []

    comp_fig, comp_ax = plt.subplots(1, 1, figsize=(6, 4))
    results = []
    for this_lr in lr_list:
        model = LinearRegressionGD(learning_rate=this_lr)
        model.set_data(X, y)
        _, _, losses = model.train_generator(epochs=epochs, update_every=1, yield_updates=False)
        results.append((this_lr, losses))
        comp_ax.plot(range(1, len(losses)+1), losses, label=f"lr={this_lr}")
    comp_ax.set_xlabel("Epoch")
    comp_ax.set_ylabel("MSE Loss")
    comp_ax.set_title("Loss curves for different learning rates")
    comp_ax.legend()
    st.pyplot(comp_fig)

# Training loop (real-time)
stop_training = False
if run_button:
    stop_training = False

# store trained predictions for export
trained_predictions = None

# Shared function to draw current state
def draw_state(X, y, m, b, losses_history=None, epoch=None):
    fig, axs = plt.subplots(1, 2 if show_loss else 1, figsize=(10, 4) if show_loss else (6,4))
    if show_loss:
        ax_data, ax_loss = axs
    else:
        ax_data = axs

    # scatter + line
    ax_data.scatter(X, y, label="Data")
    xs = np.linspace(np.min(X)-1, np.max(X)+1, 300)
    ax_data.plot(xs, m*xs + b, color="red", label=f"Fit: y={m:.3f}x + {b:.3f}")
    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    title = "Data & Fit"
    if epoch is not None:
        title += f" (epoch {epoch})"
    ax_data.set_title(title)
    ax_data.legend()

    if show_loss:
        ax_loss.plot(range(1, len(losses_history)+1), losses_history)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE Loss")
        ax_loss.set_title("Loss Curve")

    plt.tight_layout()
    return fig

# Real-time training triggered
if run_button:
    model = LinearRegressionGD(learning_rate=float(lr))
    model.set_data(X, y)

    # Use generator that yields updates every epoch (or grouped epochs)
    gen = model.train_generator(epochs=epochs, update_every=max(1, update_every))

    losses_so_far = []
    training_stopped = False

    for epoch, m_val, b_val, loss_val in gen:
        losses_so_far.append(loss_val)

        # update live parameter display
        param_text.markdown(f"- **epoch:** {epoch}  \n- **m:** {m_val:.6f}  \n- **b:** {b_val:.6f}  \n- **loss:** {loss_val:.6f}")

        # plot update
        fig = draw_state(X, y, m_val, b_val, losses_history=losses_so_far, epoch=epoch)
        plot_placeholder.pyplot(fig)

        # small sleep to make animation visible (user-controlled)
        time.sleep(speed)

        # Stop if user clicked Stop
        if stop_button:
            training_stopped = True
            st.warning("Training stopped by user.")
            break

    # final updates
    trained_predictions = pd.DataFrame({"x": X, "y_true": y, "y_pred": m_val * X + b_val})
    final_text.markdown(f"**Training finished**  \nFinal m = `{m_val:.6f}`  \nFinal b = `{b_val:.6f}`  \nFinal loss = `{loss_val:.6f}`")

    # show loss in the side-by-side (if not already)
    if show_loss:
        loss_fig = draw_state(X, y, m_val, b_val, losses_history=losses_so_far, epoch=epochs)
        loss_placeholder.pyplot(loss_fig)

# allow download of predictions
if csv_btn:
    if trained_predictions is None:
        st.error("No trained model output available. Train first.")
    else:
        towrite = io.StringIO()
        trained_predictions.to_csv(towrite, index=False)
        st.download_button("Download CSV", towrite.getvalue(), file_name="trained_predictions.csv", mime="text/csv")
