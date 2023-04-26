import torch
from torch.utils.data import DataLoader
from brainclip.model.utils.file_utils import load_dataset

class BrainCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, split_type):
        self.data = load_dataset(split_type)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, input_id_report, attention_mask_report, label = self.data[index]
        return image, input_id_report, attention_mask_report, label

class BrainCLIPDataLoader:
    def __init__(self, split_type, batch_size=2):
        self.split_type = split_type
        self.batch_size = batch_size
        self.train_dataset = BrainCLIPDataset(self.split_type)
        
    def __iter__(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return iter(train_loader)

