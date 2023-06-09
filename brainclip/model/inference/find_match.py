from brainclip.config import *
import torch
from transformers import DistilBertTokenizer
from brainclip.model.utils.file_utils import get_device
from brainclip.model.network.data_loader import BrainCLIPDataLoader
from brainclip.model.utils.processing import tokenize
import torch.nn.functional as F
from brainclip.model.network.brain_CLIP_model import BrainCLIPClassifier, ImageEncoder, TextEncoder, BrainCLIP
import json
import os

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

def sort_similarities(similarity, image_filenames, valid_reports, ground_truth):
    d = [[os.path.basename(path[0]), text, score, label] for score, path, text, label in zip(similarity, image_filenames, valid_reports, ground_truth)]
    d = sorted(d, key=lambda x: -x[2])
    d = [[*obj, rank] for obj, rank in zip(d, range(1,len(d)+1))]
    return d

def cosine_similarity(A, B):
    A_norm = F.normalize(A, p = 2, dim = -1)
    B_norm = F.normalize(B, p = 2, dim = -1)
    similarity = torch.einsum('i d, j d -> i j', A_norm, B_norm) #A_norm @ B_norm.T
    return similarity

def find_matches(model, device, query, label, n=5):

    image_embeddings, image_filenames, ground_truth, valid_reports = get_image_embeddings(model, device)
    text_embeddings = get_text_embeddings(model, device, query)

    similarity = cosine_similarity(image_embeddings, text_embeddings)
    similarity = torch.squeeze(similarity)
    similarity = similarity.cpu().numpy()
    similarity = sort_similarities(similarity, image_filenames, valid_reports, ground_truth)
    
    matches = {}
    top_matches = {}

    decode_label = lambda x: str(x.argmax(dim=1).cpu().numpy()[0])

    for match in similarity[:n]:
        top_matches[str(match[4])] = { # rank
                    "image_path":match[0],
                    "report":match[1],
                    "score":str(match[2]),
                    "label":decode_label(match[3]),
                    "rank":str(match[4]),
                    "query":query,
                    "gt":decode_label(label),
                }

    for match in similarity:
        if match[1] == query: 
            #print(f"--- {match[3]}, {match[2]}") 
            matches[str(match[4])] = { # rank
                "image_path":match[0],
                "report":match[1],
                "score":str(match[2]),
                "label":decode_label(match[3]),
                "rank":str(match[4]),
                "query":query,
                "gt":decode_label(label),
            }

    return top_matches, matches


device = get_device()
brainclip_model = BrainCLIP(ImageEncoder(), TextEncoder()).to(device) # infarct, normal, others
loaded_model = torch.load(final_model_path, map_location=device)
brainclip_model.load_state_dict(loaded_model)
brainclip_model.eval()

image_embeddings, image_filenames, ground_truth, valid_reports = get_image_embeddings(brainclip_model, device)


eval_dict = {}
top_matches = {}
for f, report, label in zip(image_filenames, valid_reports, ground_truth):
    f = os.path.basename(f[0])
    fd, d = find_matches(brainclip_model, 
             device,
             query=report,
             label=label,
             n=27
             )
    eval_dict[f] = {
        "matches":d
    }
    top_matches[f] = {
        "matches":fd
    }


#print(eval_dict)
with open(experiments_folder+"test_matches.json", "w") as f:
    json.dump(eval_dict, f)

#print(eval_dict)
with open(experiments_folder+"top_matches.json", "w") as f:
    json.dump(top_matches, f)