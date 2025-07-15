# utils/visualization_tf.py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plot_sample_images_tf(images, labels, preds=None, image_size=8):
    """
    Plots sample images, labels, and optional predictions for TensorFlow tensors.
    """
    # Convert tensors to numpy arrays
    if isinstance(images, tf.Tensor):
        images = images.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()
    if preds is not None and isinstance(preds, tf.Tensor):
        preds = preds.numpy()

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(image_size, image_size), cmap="gray")
        title = f"Label: {labels[i]}"
        if preds is not None:
            # Get the class index from one-hot or logits
            pred_class = np.argmax(preds[i])
            title += f"\nPred: {pred_class}"
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()