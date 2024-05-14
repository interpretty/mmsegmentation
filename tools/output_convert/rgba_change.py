import os
from PIL import Image


def adjust_transparency_and_color(image_path, output_folder, transparency=0.5, green_shift=1.0):
    # 打开图像
    img = Image.open(image_path)

    # 转换图像为带透明度的模式（RGBA）
    img = img.convert("RGBA")

    # 获取图像的宽度和高度
    width, height = img.size

    # 遍历图像的每个像素
    for y in range(height):
        for x in range(width):
            # 获取像素的RGBA值
            r, g, b, a = img.getpixel((x, y))

            # 降低透明度
            a = int(a * transparency)

            # 偏绿
            g = min(int(g * green_shift), 255)

            # 设置新的RGBA值
            img.putpixel((x, y), (r, g, b, a))

    # 构建输出文件名
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    # 保存偏绿的图像
    output_filename = os.path.join(output_folder, f"{name}_green{ext}")
    img.save(output_filename)

    # 保存偏蓝的图像
    blue_img = img.copy()
    for y in range(height):
        for x in range(width):
            r, g, b, a = blue_img.getpixel((x, y))
            b = min(int(b * green_shift), 255)
            blue_img.putpixel((x, y), (r, g, b, a))
    output_filename = os.path.join(output_folder, f"{name}_blue{ext}")
    blue_img.save(output_filename)

    # 保存偏红的图像
    red_img = img.copy()
    for y in range(height):
        for x in range(width):
            r, g, b, a = red_img.getpixel((x, y))
            r = min(int(r * green_shift), 255)
            red_img.putpixel((x, y), (r, g, b, a))
    output_filename = os.path.join(output_folder, f"{name}_red{ext}")
    red_img.save(output_filename)


if __name__ == "__main__":
    folder_path = r"C:\Users\WIN\Desktop\4\ARGB"
    output_folder = r"C:\Users\WIN\Desktop\4\ARGB\processed"
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(folder_path, filename)
            adjust_transparency_and_color(file_path, output_folder, transparency=1, green_shift=2.5)
