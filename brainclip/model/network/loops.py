from brainclip.config import *
from brainclip.model.utils.file_utils import update_png, get_device, load_BrainCLIP
from brainclip.model.network.brain_CLIP_model import ImageEncoder, TextEncoder, BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
import torch.nn as nn
from torch.optim import Adam
import torch

device = get_device()

train_loader = BrainCLIPDataLoader("train", batch_size=3)
val_loader = BrainCLIPDataLoader("valid", batch_size=3)

image_encoder, text_encoder = ImageEncoder(), TextEncoder()
model = BrainCLIP(image_encoder, text_encoder).to(device) # infarct, normal, others

learning_rate = 0.0001
optimizer = Adam(model.parameters(), lr=learning_rate)

num_epochs = 500
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (images, input_id_report, attention_mask_report, _, _) in enumerate(train_loader):
        images, input_id_report, attention_mask_report = images.to(device), input_id_report.to(device), attention_mask_report.to(device)
        optimizer.zero_grad()

        loss = model(images, input_id_report, attention_mask_report)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    with torch.no_grad():
        val_loss = 0.0
        for batch_idx, (images, input_id_report, attention_mask_report, _, _) in enumerate(val_loader):
            images, input_id_report, attention_mask_report = images.to(device), input_id_report.to(device), attention_mask_report.to(device)
            loss = model(images, input_id_report, attention_mask_report)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    
    print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}, temperature: {model.temperature.item():.4f}") 
    
    update_png(train_losses, val_losses, "brainclip")

    
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"{experiments_folder}brainclip_epoch_{epoch}.pt")

print("Training complete.") 
torch.save(model.state_dict(), f"{experiments_folder}brainclip_final.pt")
