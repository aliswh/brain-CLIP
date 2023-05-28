import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models.video import r3d_18
import torch.nn.functional as F
import matplotlib.pyplot as plt
from brainclip.model.utils.file_utils import get_device
from brainclip.model.network.buildingblocks import Encoder

# Define the text encoder using the pretrained 3DResNet on Kinetic400 (r3d_18)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size=400):
        super(ImageEncoder, self).__init__()
        self.embedding_size=embedding_size 

        self.resnet3d_output_size = 400 # number of classes for kinetics400
        self.resnet3d = r3d_18(weights='KINETICS400_V1')

        for param in self.resnet3d.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        x = self.resnet3d(x)
        return x
    
# 3D unet based encoder    
class ImageEncoder(Encoder):
    def __init__(self, embedding_size=400):
        self.embedding_size =embedding_size
        super(ImageEncoder, self).__init__(in_channels=3, out_channels=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(56*56*8*3, embedding_size)

    def forward(self, x):
        x = super(ImageEncoder, self).forward(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    


# pretrained DistilBERT
class TextEncoder(nn.Module):
    def __init__(self, embedding_size=400):
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
        projection_dim=256,
        dropout=0.2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        self.projection = nn.Linear(self.embedding_dim, self.projection_dim, bias = False)
        #self.relu = nn.ReLU()
        #self.fc = nn.Linear(self.projection_dim, self.projection_dim)
        #self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(self.projection_dim)
    
    def forward(self, x):
        x = self.projection(x) # projected = self.projection(x)
        #x = self.relu(projected)
        #x = self.dropout(x)
        #x = x + projected
        #x = self.layer_norm(x)
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
        self.temperature = nn.Parameter(torch.tensor(1.)) # 0.07 in paper
        self.loss_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.parameter_list = nn.ParameterList([self.temperature, self.loss_weight])
        self.device = get_device()
        self.loss_state = {"matching":[],"non_matching":[]}


    def plot_similarity(self, similarity_matrix, label):
        similarity_matrix = similarity_matrix.detach().cpu().numpy()
        label = [torch.argmax(l).detach().cpu().numpy() for l in label]
        classes = ['infarct (0)', 'tumor (1)', 'hemorrhage (2)', 'normal (3)', 'others (4)']

        plt.imshow(similarity_matrix, cmap='magma', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar()
        plt.suptitle('Similarity Matrix')
        plt.title(''.join(classes))
        plt.xlabel('Texts') 
        plt.ylabel('Images')
        plt.xticks(range(len(label)), label)
        plt.yticks(range(len(label)), label)
        plt.savefig("/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/sim.png")
        plt.close()

    def monitor_loss(self, sim):
        batch_size = sim.size(0)
        mask = torch.eye(batch_size, dtype=torch.bool, device=sim.device)
        matching_loss = sim[mask].mean()
        non_matching_loss = sim[~mask].mean()

        print_m = (1-matching_loss).detach().cpu().numpy()
        print_nm = non_matching_loss.detach().cpu().numpy()
        print(f"\tMl:{print_m:3f}, nMl:{print_nm:3f}")

        self.loss_state["matching"].append(print_m)
        self.loss_state["non_matching"].append(print_nm)

        plt.plot(range(len(self.loss_state["matching"])), self.loss_state["matching"], c="g", label='matching loss')
        plt.plot(range(len(self.loss_state["non_matching"])), self.loss_state["non_matching"], c="b", label='not matching loss')
        plt.axhline(0, c="grey")
        plt.legend()
        plt.xlabel("Per batch loss")

        plt.savefig("/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/matching_loss.png")
        plt.close()


    def forward(self, image, input_id_report, attention_mask_report, label):
        I_f = self.image_encoder(image) 
        T_f = self.text_encoder(input_id_report, attention_mask_report)

        W_i = self.image_projection(I_f)
        W_t = self.text_projection(T_f)

        I_e, T_e = map(lambda t: F.normalize(t, p = 2, dim = -1), (W_i, W_t))
        sim = torch.einsum('i d, j d -> i j', I_e, T_e) # i, j == batch_size
        self.plot_similarity(sim, label)
        sim *= self.temperature.exp()

        labels = torch.arange(I_f.size(0), device=self.device)
        I_loss = F.cross_entropy(sim, labels)
        T_loss = F.cross_entropy(sim.T, labels)

        loss = (I_loss + T_loss) / 2
        
        self.monitor_loss(sim)

        return loss


# Image-only classifier
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
        loss = F.cross_entropy
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
        x = self.braincls_fc3(x)
        #x = self.braincls_relu3(x)
        logits = self.braincls_softmax(x)
        
        if self.inference: 
            return logits
        else: return self.classification_loss(logits, label)



# Image-text classifier
class BrainCLIPClassifier(BrainCLIP):
    def __init__(self, image_encoder, text_encoder, num_classes, inference=False):
        super().__init__(image_encoder, text_encoder)
        self.num_classes = num_classes
        self.inference=inference
    
        # classification nn
        self.bcls_conv = nn.Conv1d(2, 8, kernel_size=5)
        self.bcls_relu = nn.ReLU()
        self.bcls_pool = nn.MaxPool1d(kernel_size=2)
        self.bcls_flatten = nn.Flatten() 
        self.bcls_fc = nn.Linear(1008, self.num_classes)
        self.bcls_softmax = nn.Softmax(dim=1)


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
        x = self.bcls_conv(features)
        x = self.bcls_relu(x)
        x = self.bcls_pool(x)
        x = self.bcls_flatten(x)
        x = self.bcls_fc(x)
        logits = self.bcls_softmax(x)
        
        if self.inference: return logits
        else: return self.classification_loss(logits, label)