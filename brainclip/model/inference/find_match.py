from brainclip.config import *
import torch
from transformers import DistilBertTokenizer
from brainclip.model.utils.file_utils import get_device, load_BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
from brainclip.model.utils.processing import tokenize
import torch.nn.functional as F
from brainclip.model.network.brain_CLIP_model import BrainCLIPClassifier, ImageEncoder, TextEncoder, BrainCLIP



def get_image_embeddings(model, device):
    valid_loader = BrainCLIPDataLoader("valid", batch_size=1)
    
    valid_image_embeddings = []
    valid_image_paths = []
    valid_true_class = []
    with torch.no_grad():
        for image, _, __, label, image_path in valid_loader:
            image_features = model.image_encoder(image.to(device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
            valid_image_paths.append(image_path)
            valid_true_class.append(label)
    return torch.cat(valid_image_embeddings), valid_image_paths, valid_true_class

def get_text_embeddings(model, device, query):
    encoded_query = tokenize([query])
    batch = {
        key: torch.tensor(values).to(device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            batch["input_ids"], batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    return text_embeddings

def get_embeddings(model, device, query):
    valid_loader = BrainCLIPDataLoader("valid", batch_size=1)
    
    valid_text_embeddings = []
    valid_image_embeddings = []
    valid_image_paths = []
    valid_true_class = []
    with torch.no_grad():
        for image, input_id_report, attention_mask_report, label, image_path in valid_loader:
            image_features = model.image_encoder(image.to(device))
            image_embeddings = model.image_projection(image_features)

            text_features = model.image_encoder(input_id_report.to(device), attention_mask_report.to(device))
            text_embeddings = model.text_projection(text_features)

            valid_text_embeddings.append(text_embeddings)
            valid_image_embeddings.append(image_embeddings)
            valid_image_paths.append(image_path)
            valid_true_class.append(label)
    return torch.cat(valid_image_embeddings), torch.cat(valid_text_embeddings), valid_image_paths, valid_true_class

def get_class_prediction(image_embeddings, text_embeddings):
    cat_embedding = torch.cat((image_embeddings, text_embeddings.repeat(10,1)), dim=1)
    cls_pred = brainclip_model.fc(cat_embedding)
    return cls_pred.cpu()

def correct_prediction(ground_truth, predictions):
    predictions = [torch.argmax(p) for p in predictions]
    ground_truth = [torch.argmax(p) for p in ground_truth]
    return [gt==p for gt,p in zip(ground_truth,predictions)]

def find_matches(model, device, query, n=5):

    image_embeddings, image_filenames, ground_truth = get_image_embeddings(model, device)
    text_embeddings = get_text_embeddings(model, device, query)
    
    #cls_pred = get_class_prediction(image_embeddings, text_embeddings)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [image_filenames[idx] for idx in indices]
    
    """Plot matches function""" # TODO

    print(values, indices, matches)
    #print(correct_prediction(ground_truth, cls_pred))


device = get_device()
brainclip_model = BrainCLIP(ImageEncoder(), TextEncoder()).to(device) # infarct, normal, others
brainclip_model = load_BrainCLIP(device, final_model_path, brainclip_model)

find_matches(brainclip_model, 
             device,
             query="tumor",
             n=3
             )