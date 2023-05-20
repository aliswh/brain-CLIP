import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models.video import r3d_18
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        projection_dim=300,
        dropout=0.2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.relu(projected)
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
        self.image_projection = ProjectionHead(
            embedding_dim=self.image_encoder.embedding_size)
        self.text_projection = ProjectionHead(
            embedding_dim=self.text_encoder.embedding_size)
        self.temperature = nn.Parameter(torch.tensor([0.07]), requires_grad=True) # 0.07 in paper
        self.loss_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.parameter_list = nn.ParameterList([self.temperature, self.loss_weight])
        #self.temperature = 1 #0.07
        from brainclip.model.utils.file_utils import get_device
        self.targets = torch.arange(512).to(get_device())

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def plot_similarity(self, similarity_matrix, label):
        similarity_matrix = similarity_matrix.detach().cpu().numpy()
        label = [torch.argmax(l).detach().cpu().numpy() for l in label]

        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Similarity Matrix')
        plt.xlabel('Texts') 
        plt.ylabel('Images')
        plt.xticks(range(len(label)), label)
        plt.yticks(range(len(label)), label)
        plt.savefig("/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/sim.png")
        plt.close()

    def cosine_similarity(self, A, B):
        clampled_loss_weight = torch.clamp(self.loss_weight, 0.5, 1)

        A_norm = torch.nn.functional.normalize(A, dim=1)
        B_norm = torch.nn.functional.normalize(B, dim=1)

        A_norm = clampled_loss_weight * A_norm
        B_norm = (1-clampled_loss_weight) * B_norm

        similarity = torch.matmul(A_norm, B_norm.T)
        return similarity
    
    def cosine_similarity(self, A, B):
        clampled_loss_weight = torch.clamp(self.loss_weight, 0.5, 1)

        A_norm = torch.nn.functional.normalize(A, dim=1)
        B_norm = torch.nn.functional.normalize(B, dim=1)

        #A_norm = clampled_loss_weight * A_norm
        #B_norm = (1-clampled_loss_weight) * B_norm

        similarity = A_norm @ B_norm.T
        return similarity


    def contrastive_loss(self, text_embeddings, image_embeddings, label):
        #clampled_loss_weight = torch.clamp(self.loss_weight, 0.5, 1)

        #image_embeddings = F.normalize(image_embeddings, dim=-1)
        #text_embeddings = F.normalize(text_embeddings, dim=-1)

        #logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        similarity_matrix = self.cosine_similarity(images_similarity, texts_similarity)
        #similarity_matrix = self.cosine_similarity(image_embeddings, text_embeddings)



        batch_size = similarity_matrix.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        matching_loss = similarity_matrix[mask].mean()
        non_matching_loss = similarity_matrix[~mask].mean()
        loss = ((1-matching_loss) + non_matching_loss) / 2

        print((1-matching_loss).detach().cpu().numpy(), non_matching_loss.detach().cpu().numpy())

        self.plot_similarity(similarity_matrix, label)
        
        #self.targets = F.softmax(
        #    (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        #)   

        #texts_loss = self.cross_entropy(logits, self.targets, reduction='none')
        #images_loss = self.cross_entropy(logits.T, self.targets.T, reduction='none')

        #loss = (clampled_loss_weight*images_loss + (1-clampled_loss_weight)*texts_loss) / 2 # shape: (batch_size)
        #return loss.mean()
        return loss

    

    def forward(self, image, input_id_report, attention_mask_report, label):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(input_id_report, attention_mask_report)

        image_embedding = self.image_projection(image_embedding)
        text_embedding = self.text_projection(text_embedding)

        # Calculating the Loss
        ctrs_loss = self.contrastive_loss(image_embedding, text_embedding, label)
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
        n_nodes = self.image_projection.projection_dim
        self.braincls_fc1 = nn.Linear(n_nodes, 200) # 400
        #with torch.no_grad(): self.braincls_fc1.weight.copy_(torch.Tensor([0.5]))
        self.braincls_relu1 = nn.LeakyReLU()
        #self.braincls_fc2 = nn.Linear(400, 200)
        #self.braincls_relu2 = nn.LeakyReLU()
        self.braincls_fc3 = nn.Linear(200, num_classes)
        self.braincls_relu3 = nn.LeakyReLU()
        self.braincls_softmax = nn.Softmin(dim=1)


    def classification_loss(self, cls_output, label):
        loss = nn.BCELoss()
        return loss(cls_output, label)

    def forward(self, image, input_id_report, attention_mask_report, label):
        # extract features
        image_embedding = self.image_encoder(image)
        image_embedding = self.image_projection(image_embedding)        

        # classification
        x = self.braincls_fc1(image_embedding)
        x = self.braincls_relu1(x)
        #x = self.braincls_fc2(x)
        #x = self.braincls_relu2(x)
        x = self.braincls_fc3(x)
        x = self.braincls_relu3(x)
        logits = self.braincls_softmax(x)
        
        if self.inference: 
            return logits
        else: return self.classification_loss(logits, label)




