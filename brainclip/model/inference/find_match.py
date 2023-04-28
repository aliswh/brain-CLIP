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
    with torch.no_grad():
        for image, _, __, ___, image_path in valid_loader:
            image_features = model.image_encoder(image.to(device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
            valid_image_paths.append(image_path)
    return model, torch.cat(valid_image_embeddings), valid_image_paths


def find_matches(model, device, image_embeddings, query, image_filenames, n=3):
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
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    # multiplying by 5 to consider that there are 5 captions for a single image
    # so in indices, the first 5 indices point to a single image, the second 5 indices
    # to another one and so on.
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    """Plot matches function""" # TODO

    print(values, indices, matches)


device = get_device()
model =load_BrainCLIP(device,"/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip_epoch_{epoch}.pt")
model, image_embeddings, image_filenames = get_image_embeddings(model, device)

find_matches(model, 
             device,
             image_embeddings,
             query="Acute infarct",
             image_filenames=image_filenames,
             n=3
             )