from transformers import DistilBertModel
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.optim import Adam
from brainclip.model.utils.file_utils import update_png
import torch
from torch.utils.data import DataLoader
from brainclip.model.utils.file_utils import load_dataset

class TextEncoder(nn.Module):
    def __init__(self, embedding_size=400):
        super(TextEncoder, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert.requires_grad_(False) # freeze all bert layers
        self.embedding_layer = nn.Linear(in_features=768, out_features=3)

    def forward(self, input_id_report, attention_mask_report):
        outputs = self.distilbert(input_id_report.squeeze(0), attention_mask_report)
        last_hidden_state = outputs.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        x = self.embedding_layer(CLS_token_state)
        return x
    

class BrainCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, split_type):
        self.data = load_dataset(split_type)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, input_id_report, attention_mask_report, label = self.data[index]
        return input_id_report, attention_mask_report, label

class BrainCLIPDataLoader:
    def __init__(self, split_type, batch_size=1):
        self.split_type = split_type
        self.batch_size = batch_size
        self.train_dataset = BrainCLIPDataset(self.split_type)
        
    def __iter__(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return iter(train_loader)




model = TextEncoder()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_loader = BrainCLIPDataLoader("train")

num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    for input_id_report, attention_mask_report, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(input_id_report, attention_mask_report)
        loss = criterion(outputs, labels)
        loss_history.append(loss.detach().numpy())
        print(loss)
        update_png(loss_history, prefix="text")
        loss.backward()
        optimizer.step()
