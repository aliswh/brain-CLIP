from transformers import DistilBertModel
import torch
import torch.nn as nn
from torch.optim import Adam
from brainclip.model.utils.file_utils import update_png, get_device 
import torch
from torch.utils.data import DataLoader
from brainclip.model.utils.file_utils import load_dataset
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, embedding_size=400):
        super(TextEncoder, self).__init__()
        self.embedding_size=embedding_size
        self.num_classes = 5 
        self.distilbert_output_size = 768
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert.requires_grad_(False) # freeze all bert layers
        self.embedding_layer = nn.Linear(in_features=self.distilbert_output_size, out_features=self.num_classes)

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
        image, input_id_report, attention_mask_report, label, _ = self.data[index]
        return input_id_report, attention_mask_report, label

class BrainCLIPDataLoader:
    def __init__(self, split_type, batch_size=1):
        self.split_type = split_type
        self.batch_size = batch_size
        self.train_dataset = BrainCLIPDataset(self.split_type)

    def __len__(self):
        return len(self.train_dataset)

    def __iter__(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return iter(train_loader)




device = get_device()
model = TextEncoder().to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_loader = BrainCLIPDataLoader("train")

num_epochs = 200
train_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (input_id_report, attention_mask_report, label) in enumerate(train_loader):
        # move data to device
        input_id_report, attention_mask_report, label = input_id_report.to(device), attention_mask_report.to(device), label.to(device)
        optimizer.zero_grad()

        output = model(input_id_report, attention_mask_report)
        loss = ce_loss(output, label)
        epoch_loss += loss.item()

        # backward pass and optimize
        loss.backward()
        optimizer.step()

    # log epoch loss and update plot
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    update_png(train_losses, "text") 


# ---- inference

test_loader = BrainCLIPDataLoader("valid", batch_size=2)

predictions = []
ground_truth = []
for input_id_report, attention_mask_report, label in test_loader:
    input_id_report, attention_mask_report, label = input_id_report.to(device), attention_mask_report.to(device), label.to(device)
    with torch.no_grad():
        output = model(input_id_report, attention_mask_report)
        predictions.append(output.argmax(dim=1).cpu().numpy())
        ground_truth.append(label.argmax(dim=1).cpu().numpy())

predictions = np.concatenate(predictions)
ground_truth = np.concatenate(ground_truth)

accuracy = (predictions == ground_truth).mean()
print(f"GT:{ground_truth}, \nP: {predictions}")
print(f"Accuracy: {accuracy}")
