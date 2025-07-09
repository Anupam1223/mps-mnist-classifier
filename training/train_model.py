import torch
from tqdm import tqdm
import time

def train(model, dataloader, optimizer, loss_fn, epochs=5, device='cpu', use_amp=True):
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start = time.time()

        for batch_x, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if use_amp:
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
                optimizer.step()

            total_loss += loss.item()

        duration = time.time() - start
        print(f"Epoch {epoch+1} complete in {duration:.2f}s. Avg Loss: {total_loss/len(dataloader):.4f}")
