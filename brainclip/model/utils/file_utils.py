from brainclip.config import *
from brainclip.model.utils.processing import tokenize, one_hot_encoding, preprocess_image, register_images
import os, json
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

import matplotlib.pyplot as plt
import os

def update_png(loss_history, val_loss_history=None, prefix=""):
    fig, ax = plt.subplots()
    train_line, = ax.plot(range(len(loss_history)), loss_history, color='darkgreen')
    if val_loss_history:
        valid_line, = ax.plot(range(len(val_loss_history)), val_loss_history, color='lime')

    handles, labels = [train_line], ['train loss']
    if val_loss_history:
        handles.append(valid_line)
        labels.append('valid loss')

    ax.legend(handles, labels)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.set_ylim([0, 0.5])

    plt.savefig(f"{experiments_folder}/{prefix}_loss.png")
    plt.close(fig)

    

def load_model(device, model_path, network, inference=False):
    network.load_state_dict(torch.load(model_path, map_location=device))
    if inference: return network.eval()
    else: return network

def concat_sequences(sequences_paths:list, target_path):
    nii_list = []
    for path in sequences_paths: 
        nii_list.append( sitk.ReadImage(path) )
    
    nii_list = register_images(*nii_list)

    concat_img = sitk.JoinSeries(nii_list)

    sitk.WriteImage(concat_img, target_path)


def load_dataset(split_type):
    """List of tuples like (image, report, label) in torch.Tensor format. 
    `split_type` must be in ["train","valid","test"].
    """

    assert split_type in ["train","valid","test"], ValueError


    dataset = {}
    one_hot = one_hot_encoding(['infarct', 'normal'])
    #one_hot = one_hot_encoding(['infarct', 'tumor', 'hemorrhage', 'normal', 'others'])
    json_path = os.path.join(data_folder, f"{split_type}.json")

    with open(json_path, "r") as f: 
        json_file = json.load(f) 
    
    report_batch = [[],[]] # to be tokenized later
    
    for key, value in json_file.items():
        key = int(key)
        image, report, label = value["name"], value["report"], value["label"]

        image = nib.load(image).get_fdata()

        # move modalities in first position as channels
        image = torch.Tensor(image).permute(3, 0, 1, 2)

        #image = torch.stack((image, image, image)) # TODO replace with 3 different sequences

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

