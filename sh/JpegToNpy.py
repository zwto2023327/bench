import numpy as np
from PIL import Image

# 读取JPEG图片
image_path = '/home/boot/STU/workspaces/wzx/bench/resource/blended/hello_kitty.jpeg'  # 替换为你的JPEG图片路径
image = Image.open(image_path).convert('RGB')
resized_image = image.resize((32, 32), Image.ANTIALIAS)

# 保存调整大小后的图片
resized_image_path = '/home/boot/STU/workspaces/wzx/bench/resource/blended/resized_image_32x32.jpg'  # 替换为你想要保存的图片路径
resized_image.save(resized_image_path)
# 将图片转换为NumPy数组
image_array = np.array(resized_image)

# 保存NumPy数组为.npy文件
npy_path = '/home/boot/STU/workspaces/wzx/bench/resource/blended/hello_kitty_32.npy'  # 替换为你想要保存的.npy文件路径
np.save(npy_path, image_array)