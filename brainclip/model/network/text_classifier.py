from transformers import DistilBertModel
import torch
import torch.nn as nn
from torch.optim import Adam
from brainclip.model.utils.file_utils import update_png, get_device 
import torch
from brainclip.model.network.data_loader import BrainCLIPDataLoader
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, num_classes, embedding_size=400):
        super(TextEncoder, self).__init__()
        self.embedding_size=embedding_size
        self.num_classes = num_classes
        self.distilbert_output_size = 768
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert.requires_grad_(False) # freeze all bert layers
        self.embedding_layer = nn.Linear(in_features=self.distilbert_output_size, out_features=self.num_classes)
        
        self.sigmoid = nn.Sigmoid()

    def ce_loss(self, output, label):
        if self.num_classes == 1: loss = nn.BCELoss()
        else: loss = nn.CrossEntropyLoss()
        return loss(output, label)

    def forward(self, input_id_report, attention_mask_report, label):
        outputs = self.distilbert(input_id_report.squeeze(0), attention_mask_report)
        last_hidden_state = outputs.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        x = self.embedding_layer(CLS_token_state)
        logits = self.sigmoid(x)
        return self.ce_loss(logits, label)
    
    def inference(self, input_id_report, attention_mask_report):
        outputs = self.distilbert(input_id_report.squeeze(0), attention_mask_report)
        last_hidden_state = outputs.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        x = self.embedding_layer(CLS_token_state)
        logits = self.sigmoid(x)
        return logits
    


device = get_device()
model = TextEncoder(1).to(device)
ce_loss = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_loader = BrainCLIPDataLoader("train")
val_loader = BrainCLIPDataLoader("valid")

num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (images, input_id_report, attention_mask_report, label, _) in enumerate(train_loader):
        # move data to device
        input_id_report, attention_mask_report, label = input_id_report.to(device), attention_mask_report.to(device), label.to(device)
        optimizer.zero_grad()

        loss = model(input_id_report, attention_mask_report, label)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    with torch.no_grad():
        val_loss = 0.0
        for batch_idx, (images, input_id_report, attention_mask_report, label, _) in enumerate(val_loader):
            input_id_report, attention_mask_report, label = input_id_report.to(device), attention_mask_report.to(device), label.to(device)
            loss = model(input_id_report, attention_mask_report, label)
            val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
    
    print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}") 
    
    update_png(train_losses, val_losses, "text") 
 


# ---- inference

test_loader = BrainCLIPDataLoader("test", batch_size=3)

predictions = []
ground_truth = []
for batch_idx, (images, input_id_report, attention_mask_report, labels, _) in enumerate(test_loader):
    input_id_report, attention_mask_report = input_id_report.to(device), attention_mask_report.to(device)
    with torch.no_grad():
        output = model.inference(input_id_report, attention_mask_report)
        predictions.append(torch.round(output).cpu().numpy())
        ground_truth.append(labels)

predictions = np.concatenate(predictions)
ground_truth = np.concatenate(ground_truth)

accuracy = (predictions == ground_truth).mean()
print(f"GT:{list(ground_truth)}, \nP: {predictions}")
print(f"Accuracy: {accuracy:.2f}")
