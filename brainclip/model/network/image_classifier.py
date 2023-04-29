import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.optim import Adam
from brainclip.model.utils.file_utils import update_png, get_device 
import torch
from torch.utils.data import DataLoader
from brainclip.model.utils.file_utils import load_dataset
import numpy as np

class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=400):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size 
        self.num_classes = 5 

        self.resnet3d_output_size = 400 # number of classes for kinetics400
        self.resnet3d = r3d_18(weights='KINETICS400_V1')

        # freeze all layers except last one - Linear Probing
        for param in self.resnet3d.parameters():
            param.requires_grad_(False)

        for param in self.resnet3d.layer4.parameters():
            param.requires_grad_(True)

        self.embedding_layer = nn.Linear(
            in_features=self.resnet3d_output_size,
            out_features=self.num_classes 
            )

    def forward(self, x):
        x = self.resnet3d(x)
        x = self.embedding_layer(x)
        return x
    


class BrainCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, split_type):
        self.data = load_dataset(split_type)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, input_id_report, attention_mask_report, label, img_path = self.data[index]
        return image, label

class BrainCLIPDataLoader:
    def __init__(self, split_type, batch_size=5):
        self.split_type = split_type
        self.batch_size = batch_size
        self.train_dataset = BrainCLIPDataset(self.split_type)

    def __len__(self):
        return len(self.train_dataset)
        
    def __iter__(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return iter(train_loader)



device = get_device()
model = ImageEncoder().to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_loader = BrainCLIPDataLoader("train")

num_epochs = 200
train_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (images, label) in enumerate(train_loader):
        # move data to device
        images, labels = images.to(device), label.to(device)
        optimizer.zero_grad()

        output = model(images)
        loss = ce_loss(output, label.to(device))
        epoch_loss += loss.item()

        # backward pass and optimize
        loss.backward()
        optimizer.step()

    # log epoch loss and update plot
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    update_png(train_losses, "imgclass") 


# ---- inference

test_loader = BrainCLIPDataLoader("valid", batch_size=2)

predictions = []
ground_truth = []
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        output = model(images)
        print(output)
        predictions.append(output.argmax(dim=1).cpu().numpy())
        ground_truth.append(labels.argmax(dim=1).cpu().numpy())

predictions = np.concatenate(predictions)
ground_truth = np.concatenate(ground_truth)

accuracy = (predictions == ground_truth).mean()
print(f"GT:{ground_truth}, \nP: {predictions}")
print(f"Accuracy: {accuracy}")