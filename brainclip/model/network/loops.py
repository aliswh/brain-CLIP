from brainclip.config import *
from brainclip.model.utils.file_utils import update_png
from brainclip.model.network.brain_CLIP_model import ImageEncoder, TextEncoder, BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
import torch.nn as nn
from torch.optim import Adam
import torch



num_epochs = 1
image_encoder, text_encoder = ImageEncoder(), TextEncoder()
model = BrainCLIP(image_encoder, text_encoder, num_classes=3)


learning_rate = 0.001
images_dir, reports_path = "", ""
train_loader = BrainCLIPDataLoader("train")
fine_tune = True

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

if fine_tune:
    for encoder in [model.image_encoder, model.text_encoder]:
        # Freeze all but last layer
        for name, param in model.named_parameters():
            if not 'embedding' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    for images, input_id_report, attention_mask_report, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images, input_id_report, attention_mask_report,)
        loss = criterion(outputs, labels)
        loss_history.append(loss.detach().numpy())
        update_png(loss_history, "brainclip")
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), "/datadrive_m2/alice/brain-CLIP/brainclip/model/experiments/brainclip.pt")
