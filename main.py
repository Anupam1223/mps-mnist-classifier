import torch
from data.load_mnist import get_mnist_loaders
from models.mps_classifier import MPSClassifier
from training.train_model import train
from utils.visualization import plot_sample_images

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = torch.cuda.is_available()
print(f"Using device: {device}")

# Step 1: Load resized data
train_loader, test_loader = get_mnist_loaders(batch_size=64, resize=8)

# Step 2: Create MPS model (input_dim = 8x8 = 64, bond_dim = 5)
model = MPSClassifier(input_dim=64, output_dim=10, bond_dim=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# ====== OPTIONAL: Sanity check — overfit on 1 batch ======
print("\n[Sanity Check] Trying to overfit on one batch...")

batch_x, batch_y = next(iter(train_loader))
batch_x, batch_y = batch_x.to(device), batch_y.to(device)

for i in range(100):
    optimizer.zero_grad()
    out = model(batch_x)
    loss = loss_fn(out, batch_y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, Loss: {loss.item():.4f}")

print("\n[Training Full Dataset]")
# Step 3: Train with AMP if available
train(model, train_loader, optimizer, loss_fn, epochs=5, device=device, use_amp=use_amp)

# Step 4: Evaluate and visualize
model.eval()
test_iter = iter(test_loader)
test_images, test_labels = next(test_iter)
test_images, test_labels = test_images.to(device), test_labels.to(device)

with torch.no_grad():
    logits = model(test_images)
    preds = torch.argmax(logits, dim=1)

# Step 5: Visualize predictions (move to CPU)
plot_sample_images(test_images[:5].cpu(), test_labels[:5].cpu(), preds[:5].cpu())
