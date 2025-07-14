import torch
# import tensornetwork as tn # Keep commented out
from data.load_mnist import get_mnist_loaders
from models.mps_classifier import MPSClassifier
from training.train_model import train, evaluate
from utils.visualization import plot_sample_images

# Setup device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
use_amp = False # AMP currently only for float32 on CUDA
print(f"Using device: {device}")

# --- DEBUG: Set dtype to float64 (double precision) ---
model_dtype = torch.float32
print(f"Using model dtype: {model_dtype}")
# ----------------------------------------------------

# Define your desired image size (e.g., 16 or 8)
NEW_IMAGE_SIZE = 8 # Or 8 for even faster iteration

# Step 1: Load data (28x28 MNIST)
train_loader, test_loader = get_mnist_loaders(batch_size=64, image_size=NEW_IMAGE_SIZE)

model_input_dim = NEW_IMAGE_SIZE * NEW_IMAGE_SIZE # This will be 256 or 64

# Step 2: Create MPS model
# Pass the dtype to the model constructor
model = MPSClassifier(input_dim=model_input_dim, output_dim=10, bond_dim=20, dtype=model_dtype).to(device)

# Optimizer parameters will also be in float64 if model is.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# ====== OPTIONAL: Sanity check — overfit on 1 batch ======
print("\n[Sanity Check] Trying to overfit on one batch...")

batch_x, batch_y = next(iter(train_loader))
# Crucially, convert input data to the same dtype as the model
batch_x = batch_x.to(device).to(model_dtype)
batch_y = batch_y.to(device) # Labels remain long

# Re-initialize model and optimizer for a clean sanity check run each time
# (useful if you run main.py multiple times without restarting the interpreter)
model_sanity = MPSClassifier(input_dim=model_input_dim, output_dim=10, bond_dim=5, dtype=model_dtype).to(device)
optimizer_sanity = torch.optim.Adam(model_sanity.parameters(), lr=1e-1)
loss_fn_sanity = torch.nn.CrossEntropyLoss()

for i in range(100):
    optimizer_sanity.zero_grad()
    out = model_sanity(batch_x)
    loss = loss_fn_sanity(out, batch_y)
    loss.backward()
    optimizer_sanity.step()
    print(f"Step {i}, Loss: {loss.item():.4f}")

    # --- DEBUG: Check gradients ---
    if i % 10 == 0 or i == 99:
        total_grad_norm = 0.0
        for name, param in model_sanity.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
        if total_grad_norm < 1e-12: # Check against a smaller epsilon for float64
            print(f"WARNING: Total gradient norm is extremely small at step {i}: {total_grad_norm:.12f}")
        elif i % 10 == 0:
            print(f"  Total Grad Norm at step {i}: {total_grad_norm:.12f}")
    # --- END DEBUG ---

print("\n[Training Full Dataset]")
# Step 3: Train with AMP if available (NOTE: AMP typically works with float16/bfloat16, not float64)
# If using float64 for debugging, disable AMP for full training.
train(model, train_loader, optimizer, loss_fn, epochs=5, device=device, use_amp=False)
evaluate(model, test_loader, device)

# Step 4: Evaluate and visualize
model.eval()
test_iter = iter(test_loader)
test_images, test_labels = next(test_iter)
test_images = test_images.to(device).to(model_dtype) # Convert test images to model_dtype
test_labels = test_labels.to(device)

with torch.no_grad():
    logits = model(test_images)
    preds = torch.argmax(logits, dim=1)

# Step 5: Visualize predictions (move to CPU)
plot_sample_images(test_images[:5].cpu(), test_labels[:5].cpu(), preds[:5].cpu())