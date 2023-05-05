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
        num_ftrs = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Linear(num_ftrs, self.embedding_size)
        
        # freeze all layers except last one - Linear Probing
        for param in self.resnet3d.parameters():
            param.requires_grad_(False)
        for param in self.resnet3d.fc.parameters():
            param.requires_grad_(True)   

        #for param in self.resnet3d.layer4.parameters():
        #    param.requires_grad_(True)

        self.embedding_layer = self.resnet3d.fc
        #self.embedding_layer = nn.Linear(
        #    in_features=self.resnet3d_output_size,
        #    out_features=self.embedding_size
        #    )

    def forward(self, x):
        x = self.resnet3d(x)
        x = self.embedding_layer(x)
        return x
    
class AAAImageEncoder(nn.Module):
    def __init__(self, embedding_size=4096):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size

        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.embedding_layer = nn.Linear(512, self.embedding_size)

    def forward(self, x): # x == image
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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
        #self.temperature = nn.Parameter(torch.tensor([0.07]), requires_grad=True) # 0.07 in paper
        #self.parameter_list = nn.ParameterList([self.temperature])
        self.temperature = 1

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

        #print(image_embedding.size(), text_embedding.size())

        # pad text_embedding to match image_embedding
        #max_size = max(text_embedding.size(1), image_embedding.size(1))
        #text_embedding = F.pad(text_embedding, (max_size - text_embedding.size(1), 0, 0, 0), value=0)

        # L2 normalization
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)

        # Calculating the Loss
        ctrs_loss = self.contrastive_loss(image_embedding, text_embedding)
        return ctrs_loss






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
        self.conv = nn.Conv1d(2, 8, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(8 * 98, num_classes)

    def correct_prediction(self, ground_truth, predictions):
        predictions = [torch.argmax(p) for p in predictions]
        ground_truth = [torch.argmax(p) for p in ground_truth]
        return [(gt==p).item() for gt,p in zip(ground_truth,predictions)]


    def classification_loss(self, cls_output, label):
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(cls_output, label)

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
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 8 * 98)

        logits = self.fc(x)
        softmax = nn.functional.softmax(logits, dim=1)
        cls_loss = self.classification_loss(logits, label)

        if self.inference: return softmax
        else: return cls_loss