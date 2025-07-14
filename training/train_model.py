# training/train_model.py
import torch
from tqdm import tqdm
import time

def train(model, dataloader, optimizer, loss_fn, epochs=5, device='cpu', use_amp=True):
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start = time.time()

        for batch_x, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if use_amp and device.type == 'cuda' and scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(batch_x)
                    loss = loss_fn(out, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch_x)
                loss = loss_fn(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

        duration = time.time() - start
        print(f"Epoch {epoch+1} complete in {duration:.2f}s. Avg Loss: {total_loss/len(dataloader):.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Test Accuracy: {correct / total * 100:.2f}%")