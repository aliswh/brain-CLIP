from brainclip.config import *
import os, json
import torch
import nibabel as nib
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

def update_png(loss_history, prefix=""):
    plt.plot(range(len(loss_history)), loss_history)
    plt.savefig(f"/datadrive_m2/alice/brain-CLIP/brainclip/model/network/{prefix}_loss.png")

def tokenize(text_batch):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_batch = tokenizer(text_batch, padding="max_length", return_tensors='pt')
    return encoded_batch

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

        dataset[key] = [image, None, label]

    # tokenize batch of documents
    tokenized_batch = tokenize(report_batch[1])
    
    for idx, key in enumerate(report_batch[0]):
        tokenized_report = [tokenized_batch[d][idx] for d in ["input_ids","attention_mask"]]
        dataset[key] = (dataset[key][0], *tokenized_report, dataset[key][2])

    return dataset

