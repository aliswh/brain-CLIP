from brainclip.config import *
from brainclip.model.utils.file_utils import update_png, get_device, load_BrainCLIP
from brainclip.model.network.brain_CLIP_model import ImageEncoder, TextEncoder, BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
import torch.nn as nn
from torch.optim import Adam
import torch

device = get_device()

train_loader = BrainCLIPDataLoader("train", batch_size=2)

image_encoder, text_encoder = ImageEncoder(), TextEncoder()
model = BrainCLIP(image_encoder, text_encoder).to(device) # infarct, normal, others

learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=learning_rate)

num_epochs = 300
train_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (images, input_id_report, attention_mask_report, _, _) in enumerate(train_loader):
        # move data to device
        images, input_id_report, attention_mask_report = images.to(device), input_id_report.to(device), attention_mask_report.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass and compute loss
        loss = model(images, input_id_report, attention_mask_report)
        epoch_loss += loss.item()

        # backward pass and optimize
        loss.backward()
        optimizer.step()

    # log epoch loss and update plot
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    update_png(train_losses, "brainclip")   

    
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"{experiments_folder}brainclip_epoch_{epoch}.pt")

print("Training complete.") 
torch.save(model.state_dict(), f"{experiments_folder}brainclip_final.pt")
