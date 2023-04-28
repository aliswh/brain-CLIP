import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models.video import r3d_18
import torch.nn.functional as F


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# Define the text encoder using the pretrained 3DResNet on Kinetic400 (r3d_18)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=400):
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
    def __init__(self, embedding_size=400):
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

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=200,
        dropout=0.2
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# Define the CLIP model architecture
class BrainCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, num_classes):
        super(BrainCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        self.image_projection = ProjectionHead(embedding_dim=self.image_encoder.embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=self.text_encoder.embedding_size)
        self.temperature = 1.0 # no difference
        # classification
        self.fc = nn.Linear(self.image_projection.projection_dim + self.text_projection.projection_dim, num_classes)

    def contrastive_loss(self, text_embeddings, image_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def classification_loss(self, cls_output, label):
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(cls_output, label)

    def combined_loss(self, cls_loss, ctrs_loss):
        return cls_loss + ctrs_loss

    def forward(self, image, input_id_report, attention_mask_report, label):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(input_id_report, attention_mask_report)

        # contrastive
        image_embeddings = self.image_projection(image_embedding)
        text_embeddings = self.text_projection(text_embedding)
        # classification
        cat_embeddings = torch.cat((image_embeddings, text_embeddings), dim=1)
        cls_output =  self.fc(cat_embeddings)

        # Calculating the Loss
        cls_loss = self.classification_loss(cls_output, label)
        ctrs_loss = self.contrastive_loss(image_embeddings, text_embeddings)
        return self.combined_loss(cls_loss, ctrs_loss)

