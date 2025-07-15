import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# --- From data/load_mnist.py ---
def get_mnist_loaders(batch_size=64, image_size=8):
    """
    Creates PyTorch DataLoaders for the MNIST dataset.

    Args:
        batch_size (int): The number of samples per batch.
        image_size (int): The width and height to resize the images to.

    Returns:
        A tuple of (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader

# --- From models/mps_classifier.py ---
class MPSClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=10, d_feature=2, dtype=torch.float32):
        """
        Matrix Product State (MPS) Classifier.
        
        Args:
            input_dim (int): The number of features in the input sequence (e.g., 64 for an 8x8 image).
            output_dim (int): The number of output classes (e.g., 10 for MNIST).
            bond_dim (int): The maximum bond dimension of the MPS. This controls the model's capacity.
            d_feature (int): The dimension of the feature-mapped input pixels. Default is 2.
            dtype (torch.dtype): The data type for the model's parameters.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.d_feature = d_feature
        self.dtype = dtype

        # MPS cores initialization
        self.cores = nn.ParameterList()
        gain = nn.init.calculate_gain('leaky_relu')

        for i in range(input_dim):
            if i == 0:
                core_shape = (1, d_feature, bond_dim)
            elif i == input_dim - 1:
                core_shape = (bond_dim, d_feature, bond_dim)
            else:
                core_shape = (bond_dim, d_feature, bond_dim)
            
            core = nn.Parameter(torch.empty(core_shape, dtype=self.dtype))
            nn.init.xavier_uniform_(core, gain=gain)
            self.cores.append(core)

        # Classifier layer
        self.classifier = nn.Parameter(torch.empty(output_dim, bond_dim, dtype=self.dtype))
        nn.init.xavier_uniform_(self.classifier, gain=gain)

    def feature_map(self, x):
        """ Maps input pixel values [0, 1] to a 2D feature space. """
        x_flat = x.view(x.size(0), -1)
        # Ensure the feature map output matches the model's dtype
        return torch.stack([1 - x_flat, x_flat], dim=-1).to(self.dtype)

    def forward(self, x):
        """
        Performs the forward pass by contracting the MPS with the input features.
        """
        B = x.shape[0]
        # The feature map now correctly casts the input to the model's dtype
        features = self.feature_map(x)

        # Contract each core with its corresponding feature vector
        effective_tensors = [torch.einsum('ldh,bd->blh', self.cores[i], features[:, i, :]) for i in range(self.input_dim)]
        
        # Iteratively contract the tensors in a tree-like structure
        current_tensors = effective_tensors
        while len(current_tensors) > 1:
            next_tensors = []
            # Process tensors in pairs
            for i in range(0, len(current_tensors) // 2):
                t1 = current_tensors[2 * i]
                t2 = current_tensors[2 * i + 1]
                contracted = torch.bmm(t1, t2)
                next_tensors.append(contracted)
            # If there's an odd one out, carry it over
            if len(current_tensors) % 2 == 1:
                next_tensors.append(current_tensors[-1])
            current_tensors = next_tensors

        # The final tensor after all contractions
        mps_output = current_tensors[0].squeeze(1)
        
        # Final classification layer
        logits = torch.matmul(mps_output, self.classifier.t())
        return logits

# --- From training/train_model.py ---
def train(model, dataloader, optimizer, loss_fn, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_x, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to the target device and ensure correct dtype for input
            batch_x = batch_x.to(device, dtype=model.dtype)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        duration = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete in {duration:.2f}s. Average Loss: {avg_loss:.4f}")

def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=model.dtype)
            y = y.to(device)
            
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# --- From utils/visualization.py ---
def plot_sample_images(images, labels, image_size, preds=None):
    # Ensure tensors are on CPU and converted to numpy for plotting
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if preds is not None and isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(image_size, image_size), cmap="gray")
        title = f"Label: {labels[i]}"
        if preds is not None:
            title += f"\nPred: {preds[i]}"
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---
if __name__ == '__main__':
    # Setup device: Use MPS for Apple Silicon GPUs if available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Configuration
    IMAGE_SIZE = 8
    BATCH_SIZE = 128
    BOND_DIM = 20
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    MODEL_DTYPE = torch.float32

    # 1. Load data
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    model_input_dim = IMAGE_SIZE * IMAGE_SIZE
    print(f"Data loaded. Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, Input dim: {model_input_dim}")

    # 2. Create MPS model
    model = MPSClassifier(
        input_dim=model_input_dim, 
        output_dim=10, 
        bond_dim=BOND_DIM, 
        dtype=MODEL_DTYPE
    )
    print(f"Model created with bond dimension {BOND_DIM} and dtype {MODEL_DTYPE}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. Train the model
    print("\n--- Starting Training ---")
    train(model, train_loader, optimizer, loss_fn, epochs=EPOCHS, device=device)

    # 5. Evaluate the model
    print("\n--- Evaluating on Test Set ---")
    evaluate(model, test_loader, device)

    # 6. Visualize some predictions
    print("\n--- Visualizing Sample Predictions ---")
    model.eval()
    # Get a single batch from the test loader
    test_images, test_labels = next(iter(test_loader))
    
    # Move data to device for prediction
    test_images_dev = test_images.to(device, dtype=MODEL_DTYPE)
    
    with torch.no_grad():
        logits = model(test_images_dev)
        preds = torch.argmax(logits, dim=1)

    # Plot the first 5 images, their labels, and the model's predictions
    # Tensors are moved to CPU inside the plotting function
    plot_sample_images(test_images[:5], test_labels[:5], image_size=IMAGE_SIZE, preds=preds[:5])
