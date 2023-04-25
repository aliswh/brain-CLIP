import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models.video import r3d_18

# Define the text encoder using the pretrained 3DResNet on Kinetic400 (r3d_18)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=1024):
        super(ImageEncoder, self).__init__()
        self.resnet3d = r3d_18(pretrained=True)
        self.embedding_layer = nn.Linear(in_features=512, out_features=embedding_size)

    def forward(self, x):
        x = self.resnet3d(x)
        x = self.embedding_layer(x)
        return x

# Define the text encoder using the pretrained DistilBERT
class TextEncoder(nn.Module):
    def __init__(self, embedding_size=1024):
        super(TextEncoder, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.embedding_layer = nn.Linear(in_features=768, out_features=embedding_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        x = self.embedding_layer(pooled_output)
        return x

# Define the CLIP model architecture
class BrainCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, num_classes):
        super(BrainCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.embedding_layer = nn.Linear(in_features=2048, out_features=1024)
        self.classification_layer = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, image, report, attention_mask):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(report, attention_mask)
        joint_embedding = torch.cat((image_embedding, text_embedding), dim=1)
        joint_embedding = self.embedding_layer(joint_embedding)
        logits = self.classification_layer(joint_embedding)
        class_probs = nn.functional.softmax(logits, dim=-1)
        return class_probs
    
