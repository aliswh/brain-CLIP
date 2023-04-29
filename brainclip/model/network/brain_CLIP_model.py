import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models.video import r3d_18
import torch.nn.functional as F

# Define the text encoder using the pretrained 3DResNet on Kinetic400 (r3d_18)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=400):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size 
        self.num_classes = 3 
        self.resnet3d_output_size = 400 # number of classes for kinetics400
        self.resnet3d = r3d_18(weights='KINETICS400_V1')
        self.resnet3d.requires_grad_(False)
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
    def __init__(self, image_encoder, text_encoder):
        super(BrainCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = ProjectionHead(embedding_dim=self.image_encoder.embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=self.text_encoder.embedding_size)
        self.temperature = 1.0 # no difference

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def contrastive_loss(self, text_embeddings, image_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def forward(self, image, input_id_report, attention_mask_report):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(input_id_report, attention_mask_report)

        image_embedding = self.image_projection(image_embedding)
        text_embedding = self.text_projection(text_embedding)

        # Calculating the Loss
        ctrs_loss = self.contrastive_loss(image_embedding, text_embedding)
        return ctrs_loss #self.combined_loss(cls_loss, ctrs_loss)


class BrainCLIPClassifier(nn.Module):
    def __init__(self, brainclip_model, num_classes, inference=False):
        super(BrainCLIPClassifier, self).__init__()
        self.num_classes = num_classes

        # brainCLIP
        self.inference = inference
        self.model = brainclip_model
        self.image_encoder = self.model.image_encoder
        self.text_encoder = self.model.text_encoder
        self.image_projection = ProjectionHead(embedding_dim=self.image_encoder.embedding_size)
        self.text_projection = ProjectionHead(embedding_dim=self.text_encoder.embedding_size)
        
        # classification nn
        self.fc = nn.Linear(self.image_projection.projection_dim + self.text_projection.projection_dim, num_classes)

    def correct_prediction(self, ground_truth, predictions):
        predictions = [torch.argmax(p) for p in predictions]
        ground_truth = [torch.argmax(p) for p in ground_truth]
        return [(gt==p).item() for gt,p in zip(ground_truth,predictions)]


    def classification_loss(self, cls_output, label):
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(cls_output, label)

    def forward(self, image, input_id_report, attention_mask_report, label):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(input_id_report, attention_mask_report)

        image_embedding = self.image_projection(image_embedding)
        text_embedding = self.text_projection(text_embedding)
        
        # classification
        features = torch.cat((image_embedding, text_embedding), dim=1)
        logits = self.fc(features)
        softmax = nn.functional.softmax(logits, dim=1)
        cls_loss = self.classification_loss(logits, label)

        if self.inference: return softmax
        else: return cls_loss