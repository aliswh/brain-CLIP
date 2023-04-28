from brainclip.config import *
from brainclip.model.utils.file_utils import update_png, get_device, load_BrainCLIP
from brainclip.model.network.brain_CLIP_model import ImageEncoder, TextEncoder, BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
import torch.nn as nn
from torch.optim import Adam
import torch

device = get_device()

train_loader = BrainCLIPDataLoader("train", batch_size=5)

image_encoder, text_encoder = ImageEncoder(), TextEncoder()
model = BrainCLIP(image_encoder, text_encoder, num_classes=3).to(device) # infarct, normal, others

learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=learning_rate)

num_epochs = 3
loss_history = []

for epoch in range(num_epochs):
    for images, input_id_report, attention_mask_report, labels, _ in train_loader:
        data = [d.to(device) for d in [images, input_id_report, attention_mask_report, labels]]
        optimizer.zero_grad()
        loss = model(*data)
        loss_history.append(loss.detach().cpu().numpy())
        update_png(loss_history, "brainclip")
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip_class_epoch_{epoch}.pt")


torch.save(model.state_dict(), "/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip_class_final.pt")
