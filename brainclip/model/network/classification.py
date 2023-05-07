from brainclip.config import *
from brainclip.model.utils.file_utils import update_png, get_device, load_model
from brainclip.model.network.brain_CLIP_model import BrainCLIPClassifier, ImageEncoder, TextEncoder, BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
from torch.optim import Adam, SGD
import torch
from torch.optim.lr_scheduler import MultiStepLR


device = get_device()

brainclip_model = BrainCLIP(ImageEncoder(), TextEncoder()).to(device) # infarct, normal, others
brainclip_model = load_model(device, final_model_path, brainclip_model)

model = BrainCLIPClassifier(brainclip_model, 2).to(device)

train_loader = BrainCLIPDataLoader("train", batch_size=3)
val_loader = BrainCLIPDataLoader("valid", batch_size=3)


learning_rate = 0.00001
optimizer = SGD(model.parameters(), lr=learning_rate)
scheduler = MultiStepLR(optimizer, 
                        milestones=[10,15,20,30,40,50,60,70], 
                        gamma = 0.1) 

num_epochs = 300
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (images, input_id_report, attention_mask_report, label, _) in enumerate(train_loader):
        images, input_id_report, attention_mask_report, label = images.to(device), input_id_report.to(device), attention_mask_report.to(device), label.to(device)
        optimizer.zero_grad()

        loss = model(images, input_id_report, attention_mask_report, label)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
    #scheduler.step()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    with torch.no_grad():
        val_loss = 0.0
        for batch_idx, (images, input_id_report, attention_mask_report, label, _) in enumerate(val_loader):
            images, input_id_report, attention_mask_report, label = images.to(device), input_id_report.to(device), attention_mask_report.to(device), label.to(device)
            loss = model(images, input_id_report, attention_mask_report, label)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    
    print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}, lr:{optimizer.param_groups[0]['lr']}") 
    
    update_png(train_losses, val_losses, "brainclip_cls")
    
    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip_class_epoch_{epoch}.pt")


torch.save(model.state_dict(), classification_model_path)
