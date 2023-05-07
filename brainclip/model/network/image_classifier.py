import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.optim import Adam
from brainclip.model.utils.file_utils import update_png, get_device 
import torch
from brainclip.model.network.data_loader import BrainCLIPDataLoader
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR


class ImageEncoder(nn.Module):
    def __init__(self, num_classes, embedding_size=400):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size 
        self.num_classes = num_classes

        self.resnet3d_output_size = 400 # number of classes for kinetics400
        self.resnet3d = r3d_18(weights='KINETICS400_V1')

        # freeze all layers except last one - Linear Probing
        for param in self.resnet3d.parameters():
            param.requires_grad_(True)

        for param in self.resnet3d.layer4.parameters():
            param.requires_grad_(True)

        self.embedding_layer = nn.Linear(
            in_features=self.resnet3d_output_size,
            out_features=self.num_classes 
            )
        
        self.sigmoid = nn.Sigmoid()
        
    def ce_loss(self, output, label):
        loss = nn.CrossEntropyLoss()
        return loss(output, label)

    def forward(self, x, label):
        x = self.resnet3d(x)
        x = self.embedding_layer(x)
        logits = self.sigmoid(x)
        return self.ce_loss(logits, label)
    
    def inference(self, x):
        x = self.resnet3d(x)
        x = self.embedding_layer(x)
        logits = self.sigmoid(x)
        return logits



device = get_device()
model = ImageEncoder(2).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

train_loader = BrainCLIPDataLoader("train", batch_size=16)
val_loader = BrainCLIPDataLoader("valid", batch_size=16)

num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (images, input_id_report, attention_mask_report, label, _) in enumerate(train_loader):
        images, label = images.to(device), label.to(device)
        optimizer.zero_grad()

        loss = model(images, label)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
    

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    with torch.no_grad():
        val_loss = 0.0
        for batch_idx, (images, input_id_report, attention_mask_report, label, _) in enumerate(val_loader):
            images, label = images.to(device), label.to(device)
            loss = model(images, label)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    
    print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}") 
    
    update_png(train_losses, val_losses, "imgclass") 




# ---- inference
model.eval()
test_loader = BrainCLIPDataLoader("test", batch_size=3)

predictions = []
ground_truth = []
for batch_idx, (images, input_id_report, attention_mask_report, labels, _) in enumerate(test_loader):
    images = images.to(device)
    with torch.no_grad():
        output = model.inference(images)
        predictions.append(output.argmax(dim=1).cpu().numpy())
        ground_truth.append(labels.argmax(dim=1).cpu().numpy())

predictions = np.concatenate(predictions)
ground_truth = np.concatenate(ground_truth)

accuracy = (predictions == ground_truth).mean()
print(f"GT:{list(ground_truth)}, \nP: {predictions}")
print(f"Accuracy: {accuracy:.2f}")