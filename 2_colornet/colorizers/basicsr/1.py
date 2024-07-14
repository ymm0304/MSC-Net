import sys
import os

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath('D:\\anconda\\shibie\\DDColor-master\\basicsr'))

# 获取父目录的路径
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到 Python 模块搜索路径中
sys.path.append(parent_dir)
print(sys.path)
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY