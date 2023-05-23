from brainclip.config import *
from brainclip.model.utils.file_utils import update_png, get_device, load_model
from brainclip.model.network.brain_CLIP_model import BrainCLIPClassifier, ImageEncoder, TextEncoder, BrainCLIP

from brainclip.model.network.data_loader import BrainCLIPDataLoader
from torch.optim import Adam
import torch
import numpy as np

device = get_device()

model = BrainCLIPClassifier(ImageEncoder(), TextEncoder(), 5, inference=True).to(device)
loaded_model = torch.load(classification_model_path, map_location=device)
model.load_state_dict(loaded_model, strict=False)
model.eval()

test_loader = BrainCLIPDataLoader("test", batch_size=2)

predictions = []
ground_truth = []
for images, input_id_report, attention_mask_report, labels, _ in test_loader:
    data = [d.to(device) for d in [images, input_id_report, attention_mask_report, labels]]
    with torch.no_grad():
        output = model(*data)
        predictions.append(output.argmax(dim=1).cpu().numpy())
        ground_truth.append(labels.argmax(dim=1).cpu().numpy())


predictions = np.concatenate(predictions).flatten()
ground_truth = np.concatenate(ground_truth).flatten()


accuracy = (predictions == ground_truth).mean()
print(f"GT:{ground_truth}, \nP: {predictions}")
print(f"Accuracy: {accuracy}")
