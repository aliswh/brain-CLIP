import torch
from torch.utils.data import DataLoader
from brainclip.model.utils.file_utils import load_dataset

class BrainCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, reports_path):
        self.data = load_dataset(images_dir, reports_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image, report, label = self.data[index]
        return image, report, label

class BrainCLIPDataLoader:
    def __init__(self, images_dir, reports_path, batch_size=2):
        self.images_dir = images_dir
        self.reports_path = reports_path
        self.batch_size = batch_size
        
    def __call__(self):
        train_dataset = BrainCLIPDataset(self.images_dir, self.reports_path)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_loader, len(train_dataset)
