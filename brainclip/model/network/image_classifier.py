import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.optim import Adam
from brainclip.model.utils.file_utils import update_png
import torch
from torch.utils.data import DataLoader
from brainclip.model.utils.file_utils import load_dataset

class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=200):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size 
        self.num_classes = 3 
        self.resnet3d_output_size = 400 # number of classes for kinetics400
        self.resnet3d = r3d_18(weights='KINETICS400_V1')
        self.embedding_layer = nn.Linear(
            in_features=self.resnet3d_output_size,
            out_features=self.num_classes # change to self.embedding_size
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
        image, input_id_report, attention_mask_report, label = self.data[index]
        return image, label

class BrainCLIPDataLoader:
    def __init__(self, split_type, batch_size=5):
        self.split_type = split_type
        self.batch_size = batch_size
        self.train_dataset = BrainCLIPDataset(self.split_type)
        
    def __iter__(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return iter(train_loader)




model = ImageEncoder()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_loader = BrainCLIPDataLoader("train")

num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_history.append(loss.detach().numpy())
        update_png(loss_history)
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), "/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/image_encoder.pt")
