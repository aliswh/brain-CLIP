from brainclip.config import *
import os, json
import torch
import nibabel as nib
import torch.nn.functional as F
from transformers import DistilBertTokenizer


def tokenize(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_input = tokenizer(text, return_tensors='pt')
    return encoded_input

def one_hot_encoding(labels):
    al = len(labels)
    encoding = F.one_hot(torch.arange(0, 5)% 3, num_classes=al)
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
        input_id_report, attention_mask_report = tokenize(report)
        label = one_hot[label]

        dataset[int(key)] = (image, input_id_report, attention_mask_report, label)

    
    return dataset

