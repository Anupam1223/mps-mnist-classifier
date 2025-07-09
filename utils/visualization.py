import matplotlib.pyplot as plt
import torch

def plot_sample_images(images, labels, preds=None):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if preds is not None and isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(8, 8), cmap="gray")
        title = f"Label: {labels[i]}"
        if preds is not None:
            title += f"\nPred: {preds[i]}"
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
