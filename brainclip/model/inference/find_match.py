from brainclip.config import *
import torch
from transformers import DistilBertTokenizer
from brainclip.model.utils.file_utils import get_device, load_BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
from brainclip.model.utils.processing import tokenize
import torch.nn.functional as F


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
    cat_embedding = torch.cat((image_embeddings, text_embeddings.repeat(5,1)), dim=1)
    cls_pred = model.fc(cat_embedding)
    return cls_pred.cpu()

def correct_prediction(ground_truth, predictions):
    predictions = [torch.argmax(p) for p in predictions]
    ground_truth = [torch.argmax(p) for p in ground_truth]
    return [gt==p for gt,p in zip(ground_truth,predictions)]

def find_matches(model, device, query, n=5):

    image_embeddings, image_filenames, ground_truth = get_image_embeddings(model, device)
    text_embeddings = get_text_embeddings(model, device, query)
    
    cls_pred = get_class_prediction(image_embeddings, text_embeddings)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    # multiplying by 5 to consider that there are 5 captions for a single image
    # so in indices, the first 5 indices point to a single image, the second 5 indices
    # to another one and so on.
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 1)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    """Plot matches function""" # TODO

    print(values, indices, matches)
    print(correct_prediction(ground_truth, cls_pred))


device = get_device()
model =load_BrainCLIP(device,"/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip_class_final.pt")

find_matches(model, 
             device,
             query="Acute infarct",
             n=3
             )