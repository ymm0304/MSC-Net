from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim
import torch.nn as nn
from model import FusionNetWithBoundary
import cv2
import numpy as np
import torch
from convnext_net import ConvNeXtV2
from colorization_pipline import color
model_path = r''
# 定义模型
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = FusionNetWithBoundary().to('cpu')
# weights = torch.load(model_path)
# model.load_state_dict(weights, strict=False)
model.load_state_dict(
    torch.load(r'', map_location=device)['model_state_dict'])
# model.eval()
image_path = r'D:\anconda\shibie\ca\00008.png'


def transform(path):
    image1 = cv2.imread(path)
    image1 = (image1 / 255.0).astype(np.float32)
    image1 = cv2.resize(image1, (256, 256))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = torch.from_numpy(image1.transpose((2, 0, 1))).float()
    image1 = image1.unsqueeze(0)
    return image1


img1 = transform(image_path)
with torch.no_grad():
    image_path = r''
    mask_path=r''
    img1 = transform(image_path)
    mask=transform(mask_path)
    y = model(color(img1), mask)

    final_result_tensor = y.squeeze(0)
    final_result_np = final_result_tensor.detach().numpy()
    final_result_np = (final_result_np * 255).astype(np.uint8)
    final_result_np = final_result_np.transpose((1, 2, 0))
    final_result_np = cv2.cvtColor(final_result_np, cv2.COLOR_RGB2BGR)
    cv2.imshow('Final Result', final_result_np)
    cv2.imwrite('Final Result.jpg', final_result_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
