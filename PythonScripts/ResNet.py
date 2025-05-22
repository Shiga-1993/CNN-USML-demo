import torch
import torchvision.models as models
from PIL import Image
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import tifffile
import numpy as np

args = sys.argv

n_ResNet = int(args[1])
inputfile = args[2]
output = args[3]

if n_ResNet == 18:
    model = models.resnet18(pretrained=True)
elif n_ResNet == 34:
    model = models.resnet34(pretrained=True)
elif n_ResNet == 50:
    model = models.resnet50(pretrained=True)
elif n_ResNet == 101:
    model = models.resnet101(pretrained=True)
elif n_ResNet == 152:
    model = models.resnet152(pretrained=True)
else:
    raise ValueError(f"Unsupported ResNet version: {n_ResNet}")

model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        features = torch.nn.Sequential(*list(model.children())[:-1])
        output = features(batch_t)
        output = torch.flatten(output, 1)

    return output

features = extract_features(inputfile)
print(features)

data_flattened = features.view(-1)
with open(output, 'w') as f:
    f.write(' '.join(str(value.item()) for value in data_flattened) + '\n')

