from PIL import Image

def count_pixel_colors(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 获取图像的像素值矩阵
    pixel_matrix = image.load()

    # 用字典来存储像素值和对应的计数
    pixel_count = {}

    # 遍历图像的每一个像素
    width, height = image.size
    for x in range(width):
        for y in range(height):
            # 获取像素值
            pixel = pixel_matrix[x, y]

            # 如果像素值不在字典中，则添加到字典，并设置计数为1；否则增加计数
            if pixel not in pixel_count:
                pixel_count[pixel] = 1
            else:
                pixel_count[pixel] += 1

    # 输出像素值的种类及其计数
    for color, count in pixel_count.items():
        print(f"像素值 {color} 出现 {count} 次")

    # 返回像素值的种类数
    return len(pixel_count)

# 图片路径
image_path = r"C:\PY\mmsegmentation\data\vaihingen\ann_dir\train\area1_0_0_512_512.png"  # 替换为你的图片路径

# 统计像素值的种类并输出
num_pixel_colors = count_pixel_colors(image_path)
print(f"总共有 {num_pixel_colors} 种像素值")
