import torch
import torchvision.models as models
from PIL import Image
import sys
import torch.nn as nn
from torchvision import transforms

args = sys.argv

inputfile = args[1]
output = args[2]

model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        features = model.Conv2d_1a_3x3(batch_t)
        features = model.Conv2d_2a_3x3(features)
        features = model.Conv2d_2b_3x3(features)
        features = model.maxpool1(features)
        features = model.Conv2d_3b_1x1(features)
        features = model.Conv2d_4a_3x3(features)
        features = model.maxpool2(features)
        features = model.Mixed_5b(features)
        features = model.Mixed_5c(features)
        features = model.Mixed_5d(features)
        features = model.Mixed_6a(features)
        features = model.Mixed_6b(features)
        features = model.Mixed_6c(features)
        features = model.Mixed_6d(features)
        features = model.Mixed_6e(features)
        features = model.Mixed_7a(features)
        features = model.Mixed_7b(features)
        features = model.Mixed_7c(features)
        output = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        output = torch.flatten(output, 1)

    return output

features = extract_features(inputfile)
print(features)

data_flattened = features.view(-1)
with open(output, 'w') as f:
    f.write(' '.join(str(value.item()) for value in data_flattened) + '\n')

