from brainclip.config import *
import torch
from transformers import DistilBertTokenizer
from brainclip.model.utils.file_utils import get_device
from brainclip.model.network.data_loader import BrainCLIPDataLoader
from brainclip.model.utils.processing import tokenize
import torch.nn.functional as F
from brainclip.model.network.brain_CLIP_model import BrainCLIPClassifier, ImageEncoder, TextEncoder, BrainCLIP
import json

def find_report(image_path, reports_dict):
    for key, value in reports_dict.items():
        if value["name"] == image_path[0]:
            return value["report"]

def get_image_embeddings(model, device):
    valid_loader = BrainCLIPDataLoader("test", batch_size=1)
    
    valid_image_embeddings = []
    valid_image_paths = []
    valid_true_class = []
    valid_reports = []

    with open(data_folder+"test.json","r") as f: 
        reports_dict = json.load(f)

    with torch.no_grad():
        for image, _, _, label, image_path in valid_loader:
            image_features = model.image_encoder(image.to(device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
            valid_image_paths.append(image_path)
            valid_true_class.append(label)
            valid_reports.append(find_report(image_path, reports_dict))
    return torch.cat(valid_image_embeddings), valid_image_paths, valid_true_class, valid_reports

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

def correct_prediction(ground_truth, predictions):
    predictions = [torch.argmax(p) for p in predictions]
    ground_truth = [torch.argmax(p) for p in ground_truth]
    return [gt==p for gt,p in zip(ground_truth,predictions)]

def find_matches(model, device, query, n=5):

    image_embeddings, image_filenames, ground_truth, valid_reports = get_image_embeddings(model, device)
    text_embeddings = get_text_embeddings(model, device, query)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1) 
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n)

    matches = [f"{torch.argmax(ground_truth[idx])} {image_filenames[idx]}:\n{valid_reports[idx]}\n" for idx in indices]
    
    """Plot matches function""" # TODO
    torch.set_printoptions(precision=10)

    print(values, ''.join(matches), sep="\n")
    #print(correct_prediction(ground_truth, cls_pred))


device = get_device()
brainclip_model = BrainCLIP(ImageEncoder(), TextEncoder()).to(device) # infarct, normal, others
loaded_model = torch.load(final_model_path, map_location=device)
brainclip_model.load_state_dict(loaded_model)
brainclip_model.eval()

print("\n\n\n")

find_matches(brainclip_model, 
             device,
             query="Infarct",
             n=5
             )

find_matches(brainclip_model, 
             device,
             query="No significant abnormality",
             n=5
             )