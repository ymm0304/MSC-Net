import lpips
from utils import tensor2img, img2tensor
import os
import numpy as np
import torch
from skimage import io

print(torch.cuda.is_available())
'''

https://github.com/richzhang/PerceptualSimilarity

@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}
'''
device = torch.device("cpu")


def read_image_as_rgb(file_path):
    img = io.imread(file_path)
    # Ensure the image has 3 channels (convert to RGB if not)
    if len(img.shape) == 2:  # grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 1:  # single channel image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization
import cv2

if __name__ == '__main__':
    import torch

    img0 = torch.randn(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.randn(1, 3, 64, 64)
    # d = loss_fn_alex(img0, img1).cuda()

    # print(d)

    clean_folder = 'D:/anconda/shibie/ca'
    noisy_folder = 'D:/anconda/shibie/out_sg'

    alex_losses = []
    vgg_losses = []

    clean_files = os.listdir(clean_folder)
    noisy_files = os.listdir(noisy_folder)

    for file_name in clean_files:
        clean_path = os.path.join(clean_folder, file_name)
        noisy_path = os.path.join(noisy_folder, file_name)

        if os.path.exists(noisy_path):
            clean = read_image_as_rgb(clean_path)
            noisy = read_image_as_rgb(noisy_path)
            noisy = cv2.resize(noisy, (clean.shape[1], clean.shape[0]))
            clean_tensor = img2tensor(clean).to(device)
            noisy_tensor = img2tensor(noisy).to(device)
            print(loss_fn_alex(clean_tensor, noisy_tensor))
            # print(loss_fn_vgg(clean_tensor, noisy_tensor))
            print(clean_path)
            alex_losses.append(loss_fn_alex(clean_tensor, noisy_tensor).item())
            # vgg_losses.append(loss_fn_vgg(clean_tensor, noisy_tensor).item())
        else:
            print(f'No corresponding noisy image for {file_name}')

    if alex_losses:
        avg_alex_loss = np.mean(alex_losses)
        print(f'Average Alex Loss: {avg_alex_loss}')
    else:
        print('No Alex Loss calculated.')

    if vgg_losses:
        avg_vgg_loss = np.mean(vgg_losses)
        print(f'Average VGG Loss: {avg_vgg_loss}')
    else:
        print('No VGG Loss calculated.')
