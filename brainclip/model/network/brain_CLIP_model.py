import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models.video import r3d_18

# Define the text encoder using the pretrained 3DResNet on Kinetic400 (r3d_18)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=200):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size 
        self.num_classes = 3 
        self.resnet3d_output_size = 400 # number of classes for kinetics400
        self.resnet3d = r3d_18(weights='KINETICS400_V1')
        self.embedding_layer = nn.Linear(
            in_features=self.resnet3d_output_size,
            out_features=self.embedding_size
            )

    def forward(self, x):
        x = self.resnet3d(x)
        x = self.embedding_layer(x)
        return x

# Define the text encoder using the pretrained DistilBERT
class TextEncoder(nn.Module):
    def __init__(self, embedding_size=200):
        super(TextEncoder, self).__init__()
        self.embedding_size=embedding_size
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert.requires_grad_(False) # freeze all bert layers
        self.embedding_layer = nn.Linear(in_features=768, out_features=embedding_size)

    def forward(self, input_id_report, attention_mask_report):
        outputs = self.distilbert(input_id_report.squeeze(0), attention_mask_report)
        last_hidden_state = outputs.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        x = self.embedding_layer(CLS_token_state)
        return x

# Define the CLIP model architecture
class BrainCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, num_classes):
        super(BrainCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.embedding_layer = nn.Linear(in_features=400, out_features=100)
        self.classification_layer = nn.Linear(in_features=100, out_features=num_classes)

    def forward(self, image, input_id_report, attention_mask_report):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(input_id_report, attention_mask_report)
        joint_embedding = torch.cat((image_embedding, text_embedding), dim=1)
        joint_embedding = self.embedding_layer(joint_embedding)
        logits = self.classification_layer(joint_embedding)
        class_probs = nn.functional.softmax(logits, dim=-1)
        return class_probs
    
