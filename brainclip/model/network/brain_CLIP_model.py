import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models.video import r3d_18
import torch.nn.functional as F

# Define the text encoder using the pretrained 3DResNet on Kinetic400 (r3d_18)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=512):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size 

        self.resnet3d_output_size = 400 # number of classes for kinetics400
        self.resnet3d = r3d_18(weights='KINETICS400_V1')
        
        # Use last layer (512 features) for embedding
        num_ftrs = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Linear(num_ftrs, self.embedding_size)
        
        for param in self.resnet3d.parameters():
            param.requires_grad_(True)
        # freeze all layers except last one - Linear Probing
        for param in self.resnet3d.fc.parameters():
            param.requires_grad_(True)   

        self.embedding_layer = self.resnet3d.fc


    def forward(self, x):
        x = self.resnet3d(x)
        x = self.embedding_layer(x)
        return x
    


# Define the text encoder using the pretrained DistilBERT
class TextEncoder(nn.Module):
    def __init__(self, embedding_size=512):
        super(TextEncoder, self).__init__()
        self.embedding_size=embedding_size
        self.distilbert_output_size = 768
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert.requires_grad_(False) # freeze all bert layers
        self.embedding_layer = nn.Linear(in_features=self.distilbert_output_size, out_features=embedding_size)

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
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.relu = nn.ReLU()
        #self.bnorm = nn.BatchNorm1d(num_features=3)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.relu(projected)
        #x = self.bnorm(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x

# Define the CLIP model architecture
class BrainCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super(BrainCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = ProjectionHead(
            embedding_dim=self.image_encoder.embedding_size,
            projection_dim=self.image_encoder.embedding_size)
        self.text_projection = ProjectionHead(
            embedding_dim=self.text_encoder.embedding_size,
            projection_dim=self.text_encoder.embedding_size)
        self.temperature = nn.Parameter(torch.tensor([0.07]), requires_grad=True) # 0.07 in paper
        self.loss_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.parameter_list = nn.ParameterList([self.temperature, self.loss_weight])
        #self.temperature = 1 #0.07

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

        loss = (self.loss_weight*images_loss + (1-self.loss_weight)*texts_loss) / logits.size(0) # shape: (batch_size)
        return loss.mean()

    def forward(self, image, input_id_report, attention_mask_report):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(input_id_report, attention_mask_report)

        image_embedding = self.image_projection(image_embedding)
        text_embedding = self.text_projection(text_embedding)

        # Calculating the Loss
        ctrs_loss = self.contrastive_loss(image_embedding, text_embedding)
        return ctrs_loss

"""




class BrainCLIPClassifier(nn.Module):
    def __init__(self, brainclip_model, num_classes, inference=False):
        super(BrainCLIPClassifier, self).__init__()
        self.num_classes = num_classes

        # brainCLIP
        self.inference = inference
        self.model = brainclip_model

        self.image_encoder = self.model.image_encoder
        self.text_encoder = self.model.text_encoder

        self.image_projection = ProjectionHead(
            embedding_dim=self.image_encoder.embedding_size,
            projection_dim=self.image_encoder.embedding_size)
        self.text_projection = ProjectionHead(
            embedding_dim=self.text_encoder.embedding_size,
            projection_dim=self.text_encoder.embedding_size)
        
        # classification nn
        self.conv = nn.Conv1d(2, 8, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten() 
        self.fc = nn.Linear(2032, self.num_classes)
        self.softmax = nn.Softmax(dim=1)


    def classification_loss(self, cls_output, label):
        loss = nn.CrossEntropyLoss()
        return loss(cls_output, label)

    def forward(self, image, input_id_report, attention_mask_report, label):
        # extract features
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(input_id_report, attention_mask_report)

        image_embedding = self.image_projection(image_embedding)
        text_embedding = self.text_projection(text_embedding)
        
        # stack together
        features = torch.stack((image_embedding, text_embedding), dim=1)

        # classification
        x = self.conv(features)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        logits = self.softmax(x)
        
        if self.inference: return logits
        else: return self.classification_loss(logits, label)

"""

class BrainCLIPClassifier(BrainCLIP):
    def __init__(self, image_encoder, text_encoder, num_classes, inference=False):
        super().__init__(image_encoder, text_encoder)
        self.num_classes = num_classes
        self.inference=inference

        # classification nn
        n_nodes = self.image_encoder.embedding_size
        self.braincls_fc1 = nn.Linear(n_nodes, 2) # 400
        #with torch.no_grad(): self.braincls_fc1.weight.copy_(torch.Tensor([0.5]))
        #self.braincls_relu1 = nn.ReLU()
        #self.braincls_fc2 = nn.Linear(400, 200)
        #self.braincls_relu2 = nn.ReLU()
        #self.braincls_fc3 = nn.Linear(200, num_classes)
        #self.braincls_relu3 = nn.ReLU()
        self.braincls_softmax = nn.Softmax(dim=1)


    def classification_loss(self, cls_output, label):
        loss = nn.BCELoss()
        return loss(cls_output, label)

    def forward(self, image, input_id_report, attention_mask_report, label):
        # extract features
        image_embedding = self.image_encoder(image)
        image_embedding = self.image_projection(image_embedding)        

        # classification
        x = self.braincls_fc1(image_embedding)
        #x = self.braincls_relu1(x)
        #x = self.braincls_fc2(x)
        #x = self.braincls_relu2(x)
        #x = self.braincls_fc3(x)
        #x = self.braincls_relu3(x)
        logits = self.braincls_softmax(x)
        
        if self.inference: 
            return logits
        else: return self.classification_loss(logits, label)




