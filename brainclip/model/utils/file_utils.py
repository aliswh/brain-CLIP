from brainclip.config import *
from brainclip.model.network.brain_CLIP_model import ImageEncoder, TextEncoder, BrainCLIP
from brainclip.model.utils.processing import tokenize, one_hot_encoding
import os, json
import torch
import nibabel as nib
import matplotlib.pyplot as plt

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def update_png(loss_history, prefix=""):
    plt.plot(range(len(loss_history)), loss_history)
    plt.savefig(f"/datadrive_m2/alice/brain-CLIP/brainclip/model/network/{prefix}_loss.png")

def load_BrainCLIP(device, model_path):
    image_encoder, text_encoder = ImageEncoder(), TextEncoder()
    model = BrainCLIP(image_encoder, text_encoder, num_classes=3).to(device) # infarct, normal, others
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.eval()


def load_dataset(split_type):
    """List of tuples like (image, report, label) in torch.Tensor format. 
    `split_type` must be in ["train","valid","test"].
    """

    assert split_type in ["train","valid","test"], ValueError


    dataset = {}
    one_hot = one_hot_encoding(['infarct', 'normal', 'others'])
    json_path = os.path.join(data_folder, f"{split_type}.json")

    with open(json_path, "r") as f: 
        json_file = json.load(f) 
    
    report_batch = [[],[]] # to be tokenized later
    
    for key, value in json_file.items():
        key = int(key)
        image, report, label = value["name"], value["report"], value["label"]

        image = nib.load(image).get_fdata()
        image = torch.Tensor(image)
        image = torch.stack((image, image, image)) # TODO replace with 3 different sequences

        report_batch[0].append(key) 
        report_batch[1].append(report)

        label = one_hot[label]

        dataset[key] = [image, None, label, value["name"] ] # return also image path

    # tokenize batch of documents
    tokenized_batch = tokenize(report_batch[1])
    
    for idx, key in enumerate(report_batch[0]):
        tokenized_report = [tokenized_batch[d][idx] for d in ["input_ids","attention_mask"]]
        dataset[key] = (dataset[key][0], *tokenized_report, dataset[key][2], dataset[key][3])

    return dataset

