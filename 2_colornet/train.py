from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim
import torch.nn as nn
from model import FusionNetWithBoundary
import cv2
import numpy as np
from colorization_pipline import color

def transform(path):
    image1 = cv2.imread(path)
    image1 = (image1 / 255.0).astype(np.float32)
    image1 = cv2.resize(image1, (256, 256))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = torch.from_numpy(image1.transpose((2, 0, 1))).float()
    # image1 = image1.unsqueeze(0)
    return image1


class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir,target, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_dir = target
        self.transform = transform

        self.image_files = os.listdir(image_dir)
        self.mask_files = os.listdir(mask_dir)
        self.target_files = os.listdir(target)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # print(idx)
        image_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        target_name = self.target_files[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        target_path = os.path.join(self.target_dir, target_name)
        image_1 = cv2.imread(image_path)
        mask = Image.open(mask_path)
        image_2 = cv2.imread(mask_path)
        image = transform(image_path)
        target = transform(target_path)
        mask= transform(mask)
        return image, target,mask


# 定义数据集和数据加载器

image_dir = r''
gt_dir = r''
mask_dir = r''

dataset = MyDataset(color(image_dir), mask_dir, gt_dir,transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
import torch

# 定义模型
model = FusionNetWithBoundary().cuda()

torch.cuda.empty_cache()


def save_model(model, epoch):
    # 这里保存模型的逻辑，例如使用torch.save
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 还可以保存优化器状态等
    }, f'model_checkpoint_{epoch}.pth')


criterion_image = nn.MSELoss()

criterion = nn.L1Loss()  # 图像重建损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
weight_decay_factor = 0.001
num_epochs = 100  # 假设您想要运行100个epoch
best_loss = float('inf')  # 初始化一个很大的值，用于跟踪最佳损失
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (image, target, mask) in enumerate(dataloader):
        optimizer.zero_grad()
        print("image.shape",image.shape)
        # print(image.size(),boundary.size(),target.size())
        image = image.cuda()
        target = target.cuda()
        # print(image.size(), boundary.size())
        output = model(image,maks)
        # print(output.size(), target.size())
        # 计算图像重建损失
        reconstruction_loss = criterion_image(output, target) + criterion(output, target)

        # 总体损失为图像重建损失加上边界损失
        # loss = reconstruction_loss + boundary_loss
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param) ** 2
        l2_reg *= weight_decay_factor

        # 总体损失为图像重建损失加上L2正则化惩罚项
        loss = reconstruction_loss + l2_reg
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # running_loss += loss.item()

        # 计算每个epoch的平均损失
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item()}')

    avg_loss = running_loss / len(dataloader)

    # 打印每个epoch的结果
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 如果当前epoch的损失是最低的，保存模型
    # if avg_loss < best_loss:
    #     best_loss = avg_loss
    #     save_model(model, epoch)

    # 每50个epoch保存一次模型
    if (epoch + 1) % 20 == 0:
        save_model(model, epoch)

    # 训练结束后，保存最终模型（可选）
save_model(model, num_epochs - 1)
#
# for name, layer in model.named_children():

