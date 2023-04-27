from brainclip.config import *
import os, json
import torch
import nibabel as nib
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

def update_png(loss_history):
    plt.plot(range(len(loss_history)), loss_history)
    plt.savefig("/datadrive_m2/alice/brain-CLIP/brainclip/model/network/loss.png")

def pad_tensor(t):
    max_len = 480 # TODO
    t = F.pad(t, (0, max_len - t.size(0)), mode='constant', value=0)
    return t

def tokenize(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_input = tokenizer(text, return_tensors='pt')
    input_id, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]
    input_id = pad_tensor(input_id)
    attention_mask = pad_tensor(attention_mask)
    return input_id, attention_mask

def one_hot_encoding(labels):
    al = len(labels)
    encoding = F.one_hot(torch.arange(0, 5)% 3, num_classes=al).float()
    return {l:e for l,e in zip(labels,encoding)}


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
    
    
    for key, value in json_file.items():
        image, report, label = value["name"], value["report"], value["label"]

        image = nib.load(image).get_fdata()
        image = torch.Tensor(image)
        image = torch.stack((image, image, image)) # TODO replace with 3 different sequences

        input_id_report, attention_mask_report = tokenize(report)
        label = one_hot[label]

        dataset[int(key)] = (image, input_id_report, attention_mask_report, label)

    return dataset

