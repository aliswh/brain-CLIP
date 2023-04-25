from brainclip.model.network.brain_CLIP_model import ImageEncoder, TextEncoder, BrainCLIP
from brainclip.model.network.data_loader import BrainCLIPDataLoader
import torch.nn as nn
from torch.optim import Adam

num_epochs = 3
image_encoder, text_encoder = ImageEncoder(), TextEncoder()
model = BrainCLIP(image_encoder, text_encoder)

learning_rate = 0.001
images_dir, reports_path = "", ""
train_loader, total_step = BrainCLIPDataLoader(images_dir, reports_path)
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

for epoch in range(num_epochs):
    for i, (images, reports, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images, reports)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
