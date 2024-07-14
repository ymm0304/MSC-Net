import torch
import torch.nn as nn
from cont import ConvNeXt, convnext_small  # 假设ConvNeXt模型可以通过这种方式导入
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    def __init__(self, input_channels, boundary_channels, output_channels, spatial_size=1):
        super(FusionLayer, self).__init__()
        # 注意：这里的input_channels和boundary_channels应该是相同的，因为它们都是4
        # spatial_size定义了平铺后的空间大小，这里设为1仅作为示例
        self.conv1 = nn.Conv2d(1536, 768, kernel_size=1)  # 使用1x1卷积
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image_features):
        fused_features = self.conv1(image_features)

        # 应用ReLU激活函数
        fused_features = self.relu(fused_features)

        return fused_features


class Encoder(nn.Module):

    def __init__(self, encoder_name):
        super().__init__()
        if encoder_name == 'convnext-t' or encoder_name == 'convnext':
            self.arch = ConvNeXt()
            self.load('DDColor-master/pretrain/convnext_tiny_22k_224.pth')
        elif encoder_name == 'convnext-s':
            # self.arch = convnext_small()
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
            self.load(r'D:\anconda\shibie\DDColor-master\pretrain\convnext_large_22k_224.pth')
        elif encoder_name == 'convnext-b':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
            self.load('DDColor-master/ppretrain/convnext_base_22k_224.pth')
        elif encoder_name == 'convnext-l':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
            self.load('E:\\ddcolor\\ddco\\DDColor-master\\pretrain\\convnext_large_22k_224.pth')
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.arch(x)

    def load(self, path):
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        if not path:
            logger.info("No checkpoint found. Initializing model from scratch")
            return
        logger.info("[Encoder] Loading from {} ...".format(path))
        checkpoint = torch.load(path)
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        incompatible = self.arch.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            msg = "Some model parameters or buffers are not found in the checkpoint:\n"
            msg += str(incompatible.missing_keys)
            logger.warning(msg)
        if incompatible.unexpected_keys:
            msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
            msg += str(incompatible.unexpected_keys)
            logger.warning(msg)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)

        value = self.value(x).view(batch_size, -1, height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out


class FusionNetWithBoundary(nn.Module):
    def __init__(self):
        super(FusionNetWithBoundary, self).__init__()

        # 加载预训练的ConvNeXt模型权重
        pretrained_weights_path = r'E:\ddcolor\ddco\DDColor-master\pretrain\convnext_large_22k_224.pth'

        self.encoder = Encoder('convnext-s')
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.decoder = Decoder()
        self.attention=SelfAttention()
        # 融合层定义...
        # 边界信息卷积层
        self.boundary_conv = nn.Conv2d(1, 64, 3, padding=1)  # 假设边界图像是单通道的
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.head_cls = nn.Linear(768, 4)

    def load(self, path):
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        if not path:
            logger.info("No checkpoint found. Initializing model from scratch")
            return
        logger.info("[Encoder] Loading from {} ...".format(path))
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        incompatible = self.arch.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            msg = "Some model parameters or buffers are not found in the checkpoint:\n"
            msg += str(incompatible.missing_keys)
            logger.warning(msg)
        if incompatible.unexpected_keys:
            msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
            msg += str(incompatible.unexpected_keys)
            logger.warning(msg)

    def normalize(self, img):
        mean = torch.mean(img)
        std = torch.std(img)
        return (img - mean) / std

    def denormalize(self, img1):
        mean = torch.mean(img1)
        std = torch.std(img1)
        return img1 * std + mean

    def forward(self, image,mask):
        # Encoder部分处理image_A和image_B...
        # 融合层处理...
        # print(image)
        # print(self.denormalize((image)))
        image1 = self.normalize(image)
        # print("enorm:",image1)
        # print("denorm:", self.denormalize(image,image1))
        encoder_out1 = self.encoder(image1)
        # print(encoder_out1.shape)
        # x_tripled = boundary.repeat(1, 3, 1, 1)
        # print(image.size())
        encoder_out2 = self.encoder(mask)
        encoder_out2=self.attention(encoder_out2)
        # print(encoder_out2.shape)

        # fused_features1 = self.decoder(encoder_out2)
        # print("decoder", fused_features1.shape)

        # concatenated = torch.cat((encoder_out1, encoder_out2), dim=1)
        # fused_features = self.fusion_layer(concatenated)
        # print("fused_features", fused_features)

        fused_features1 = self.decoder(encoder_out1)
        print(fused_features1.shape,encoder_out2.shape)
        # 将边界信息传递给模型

        # print("boundary_features", boundary_features.shape)
        # 将边界信息与编码器的输出拼接
        # fused_features = torch.cat((fused_features, boundary_features), dim=1)

        # boundary_features = self.boundary_conv(boundary)
        import torch.nn.functional as F
        # 计算边界的梯度

        # 计算梯度幅值的平方和作为边界损失
        boundary_loss = 1
        # fused_features1 = self.denormalize(fused_features1)
        # print("fused_features", encoder_out1)
        # print(x.shape, output.shape)
        fused_features1 = self.denormalize(fused_features1)
        return fused_features1


class Decoder(nn.Module):
    def __init__(self):
        output_channels = 3
        super(Decoder, self).__init__()

        # 接下来的转置卷积层配置根据您的需要进行调整
        self.deconv1 = nn.ConvTranspose2d(in_channels=768, out_channels=384, kernel_size=4, stride=4, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=6, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=6, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=96, out_channels=3, kernel_size=7, stride=2, padding=1,
                                          output_padding=1)

        self.activation = nn.ReLU()

    def forward(self, convnext_output):
        # 展平encoder的输出

        x = convnext_output
        # 通过转置卷积层进行上采样
        x1 = self.activation(self.deconv1(x))
        x2 = self.activation(self.deconv2(x1))
        x3 = self.activation(self.deconv3(x2))
        x4 = torch.sigmoid(self.deconv4(x3))

        # 此时x4的形状应该是(16, 3, 256, 256)，但您需要通过调整转置卷积的stride和padding来确保这一点
        return x4


def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        val_range = max_val - min_val

    if window is None:
        real_size = min(window_size, img1.size(2), img2.size(2))
        window = create_window(real_size, img1.size(1)).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
