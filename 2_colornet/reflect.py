import cv2
import numpy as np


def restore_highlight(removed_highlight_image, original_image, highlight_mask, alpha=0.5):
    """
    将高光信息恢复到被消除高光的图像上

    参数：
        removed_highlight_image (numpy.ndarray): 经过消除高光的图像
        original_image (numpy.ndarray): 原始图像
        highlight_mask (numpy.ndarray): 高光区域的掩码（1表示高光区域，0表示其他区域）
        alpha (float): 加权融合的权重参数，范围在0到1之间，默认为0.5

    返回：
        restored_image (numpy.ndarray): 恢复了高光信息的图像
    """
    # 确保图像和掩码尺寸相同
    assert removed_highlight_image.shape == original_image.shape == highlight_mask.shape

    # 使用加权融合方法将高光信息恢复到被消除高光的图像上
    restored_image = alpha * removed_highlight_image + (1 - alpha) * original_image

    # 将高光区域的像素值设为原始图像中对应位置的像素值
    restored_image[np.where(highlight_mask == 1)] = original_image[np.where(highlight_mask == 1)]

    return restored_image


# 加载被消除高光的图像、原始图像和高光掩码
removed_highlight_image = cv2.imread(r"")
original_image = cv2.imread(r"")
highlight_mask = cv2.imread(r"", cv2.IMREAD_GRAYSCALE)
highlight_mask = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2RGB)

print("Removed Highlight Image Shape:", removed_highlight_image.shape)
print("Original Image Shape:", original_image.shape)
print("Highlight Mask Shape:", highlight_mask.shape)

# 将图像转换为浮点数类型
removed_highlight_image = removed_highlight_image.astype(np.float32) / 255.0
original_image = original_image.astype(np.float32) / 255.0

# 调用恢复高光信息的函数
restored_image = restore_highlight(removed_highlight_image, original_image, highlight_mask)

# 将恢复的图像保存到文件
cv2.imwrite("restored_image.png", (restored_image * 255).astype(np.uint8))
