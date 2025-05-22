import torch
import torchvision.models as models
from PIL import Image
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import tifffile
import numpy as np
args = sys.argv

n_DenseNet = int(args[1])
inputfile = args[2]
output = args[3]

if n_DenseNet == 121:
    model = models.densenet121(pretrained=True)
elif n_DenseNet == 169:
    model = models.densenet169(pretrained=True)
elif n_DenseNet == 201:
    model = models.densenet201(pretrained=True)
elif n_DenseNet == 161:
    model = models.densenet161(pretrained=True)
model.eval()  # 推論モードに設定

# 画像の前処理を定義（DenseNetに適した形式に変換）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 特徴量抽出関数
def extract_features(image_path):
    # 画像の読み込みと前処理
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # 特徴量の抽出
    with torch.no_grad():
        # 最後の全結合層を除いたモデルを使用
        features = torch.nn.Sequential(*list(model.features.children()))
        output = features(batch_t)
        # Global average pooling
        output = nn.functional.adaptive_avg_pool2d(output, (1, 1))
        output = torch.flatten(output, 1)

    return output

# 使用例
# この部分は実際の画像ファイルのパスに置き換えてください
features = extract_features(inputfile)
print(features)  # 特徴量ベクトルを表示
# Reshape the tensor to a one-dimensional array
data_flattened = features.view(-1)
# Save to text file
with open(output, 'w') as f:
    # 各要素を空白で区切って一行に書き出す
    f.write(' '.join(str(value.item()) for value in data_flattened) + '\n')
