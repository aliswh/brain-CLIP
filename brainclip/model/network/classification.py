from brainclip.config import *
from brainclip.model.utils.file_utils import update_png, get_device, load_BrainCLIP
from brainclip.model.network.brain_CLIP_model import BrainCLIPClassifier, ImageEncoder, TextEncoder, BrainCLIP

from brainclip.model.network.data_loader import BrainCLIPDataLoader
from torch.optim import Adam
import torch

device = get_device()

brainclip_model = BrainCLIP(ImageEncoder(), TextEncoder()).to(device) # infarct, normal, others
brainclip_model = load_BrainCLIP(device, final_model_path, brainclip_model)

model = BrainCLIPClassifier(brainclip_model, 3).to(device)

train_loader = BrainCLIPDataLoader("train", batch_size=5)

learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=learning_rate)

num_epochs = 20
loss_history = []

for epoch in range(num_epochs):
    for images, input_id_report, attention_mask_report, labels, _ in train_loader:
        data = [d.to(device) for d in [images, input_id_report, attention_mask_report, labels]]
        optimizer.zero_grad()
        loss = model(*data)
        loss_history.append(loss.detach().cpu().numpy())
        update_png(loss_history, "classification")
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip_class_epoch_{epoch}.pt")


torch.save(model.state_dict(), "/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip_class_final.pt")
