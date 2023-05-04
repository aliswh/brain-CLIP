import torch
from torch.utils.data import DataLoader
from brainclip.model.utils.file_utils import load_dataset
from brainclip.model.utils.transforms import apply_transform, get_transforms

class BrainCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, split_type):
        self.split_type = split_type
        self.data = load_dataset(split_type)
        self.transforms = get_transforms()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, input_id_report, attention_mask_report, label, image_path = self.data[index]
        if self.split_type=="train": image = apply_transform(image, self.transforms)
        return image, input_id_report, attention_mask_report, label, image_path

class BrainCLIPDataLoader:
    def __init__(self, split_type, batch_size=1):
        self.split_type = split_type
        self.batch_size = batch_size
        self.shuffle = True if split_type == "train" else False
        self.train_dataset = BrainCLIPDataset(self.split_type)
        
    def __len__(self):
        return len(self.train_dataset)

    def __iter__(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return iter(train_loader)

