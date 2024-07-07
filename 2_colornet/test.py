from colornet import *
import matplotlib.pyplot as plt
from colorizers import *
import torch.nn as nn


class SIGGRAPHGeneratorModified(SIGGRAPHGenerator1):
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super(SIGGRAPHGeneratorModified, self).__init__(norm_layer, classes)
        # 由于输入现在是4通道的（包括L通道和高光mask），我们需要修改第一个卷积层以接受4通道输入
        self.model1[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, input_A, input_B):
        # 由于在train函数中已经处理了输入，这里不需要额外的输入处理
        return super(SIGGRAPHGeneratorModified, self).forward(input_A, input_B)


def resize_img(img, HW, resample):
    # 检查img是否为Tensor
    if isinstance(img, torch.Tensor):
        # 确保Tensor在CPU上
        img = img.cpu()
        # 转换为NumPy数组
        img = img.numpy()
        # 转换为HWC格式，如果需要
        if img.ndim == 3 and img.shape[0] in {1, 3}:
            img = img.transpose(1, 2, 0)
        if img.shape[2] == 1:
            # 如果图像是单通道的，转换为灰度图
            img = img.squeeze(-1)
    # 如果img是float类型的NumPy数组，先转换为uint8（假设图像数据范围是[0, 1]）
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)

    # 使用PIL进行图像尺寸调整
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize((HW[1], HW[0]), resample=resample)
    # 可以选择是否转换回Tensor
    return np.asarray(img_resized)


from torchvision.io import read_image
from torchvision.transforms.functional import resize, to_tensor
from skimage import color
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ColorizationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),  # 将图像缩放到统一尺寸
        ])
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)  # 假设mask与图像文件名相同
        HW = (256, 256)
        resample = 3
        mask1 = load_gray_img(mask_path)
        mask_rs = resize_img(mask1, HW=HW, resample=resample)
        mask = self.transform(mask_rs)

        img_rgb_orig = load_img(img_path)
        img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

        img_lab_orig = color.rgb2lab(img_rgb_orig)
        img_lab_rs = color.rgb2lab(img_rgb_rs)

        img_l_orig = img_lab_orig[:, :, 0]
        img_l_rs = img_lab_rs[:, :, 0]

        # 仅保留a和b通道
        img_ab = img_lab_rs[:, :, 1:3]
        tens_orig_l = torch.Tensor(img_l_orig)[None, :, :]
        tens_rs_l = torch.Tensor(img_l_rs)[None, :, :]
        tens_rs_ab = torch.Tensor(img_ab)[:, :]
        tens_rs_ab = tens_rs_ab.permute(2, 0, 1)
        return tens_rs_l, tens_rs_ab, mask  # 归一化mask


device = torch.device('cuda:0')




def all():
    pretrained_weights = torch.load('')
    model1 = SIGGRAPHGeneratorModified()
    model1.load_state_dict(pretrained_weights)
    # 初始化你的模型
    model = SIGGRAPHGenerator()
    pretrained_weights1 = torch.load('')
    model.load_state_dict(pretrained_weights1)
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions

    img_folder_path = ''
    mask_folder_path = ''
    output_folder_path = ''

    # 遍历文件夹内的所有图片
    for filename in os.listdir(img_folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(img_folder_path, filename)
            mask_path = os.path.join(mask_folder_path, filename)

            # 加载并处理图像和掩码
            img = load_img(img_path)
            mask = load_gray_img(mask_path)

            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
            mask_rs = preprocess_mask(mask, HW=(256, 256))
            mask_rs = mask_rs.repeat(1, 2, 1, 1)
            tens_l_rs = tens_l_rs.repeat(1, 2, 1, 1)

            # 使用模型进行颜色化处理
            out_img_siggraph17 = postprocess_tens(tens_l_orig, model1(tens_l_rs, mask_rs).cpu())

            # 保存处理后的图像
            output_path = os.path.join(output_folder_path, filename.replace('.png', '_siggraph17.png'))
            plt.imsave(output_path, out_img_siggraph17)
            print(f'Processed and saved: {output_path}')

all()#测试整个文件夹图片