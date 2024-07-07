import os
import cv2

def convert_to_3_channel(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取图像
        img = cv2.imread(input_path)

        # 如果图像是单通道的，将其转换为3通道
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 保存标准化后的图像
        cv2.imwrite(output_path, img)

        # 保存标准化后的图像
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""

    convert_to_3_channel(input_folder, output_folder)
