import cv2
import os

# 矩形区域参数
a1, b1, c1, d1 = 78,	145,	256,	295  # 矩形1参数,替换为实际值或None
a2, b2, c2, d2 = None, None, None, None  # 矩形2参数,替换为实际值或None
a3, b3, c3, d3 = None, None, None, None  # 矩形3参数,替换为实际值或None

# 绘制矩形的颜色 (R,G,B)
rect_color = (220, 160, 160)  # 深紫色

# 输入目录
input_dirs = [
    "C:\\PY\\mmsegmentation\\checkpoints\\potsdam\\unetformerwr_80k\\test",
    "C:\\PY\\mmsegmentation\\data\\potsdam\\ann_dir\\test",
    "C:\\Users\\WIN\\Desktop\\Desktop\\potsdam\\unetformer",
    "C:\\Users\\WIN\\Desktop\\Desktop\\potsdam\\danet",
    "C:\\Users\\WIN\\Desktop\\Desktop\\potsdam\\deeplabv3plus",
    "C:\\Users\\WIN\\Desktop\\Desktop\\potsdam\\ocrnet",
    "C:\\Users\\WIN\\Desktop\\Desktop\\potsdam\\segformer"
]

# 输出目录
output_dir = "C:\\Users\\WIN\\Desktop\\potsdam"

# 图像名称前缀
pic_name = "2_13_5120_3072_5632_3584"

for input_dir in input_dirs:
    # 构建输入文件名
    if input_dir == input_dirs[0]:
        img_name = pic_name + ".png"
    elif input_dir == input_dirs[1]:
        img_name = pic_name + "_converted.png"
    else:
        img_name = pic_name + ".png"

    # 读取图像
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    # 绘制矩形
    if a1 is not None:
        cv2.rectangle(img, (a1, b1), (c1, d1), rect_color, 4)
    if a2 is not None:
        cv2.rectangle(img, (a2, b2), (c2, d2), rect_color, 4)
    if a3 is not None:
        cv2.rectangle(img, (a3, b3), (c3, d3), rect_color, 4)

    # 构建输出文件名
    if input_dir == input_dirs[1]:
        output_filename = "GT_" + img_name
    else:
        output_filename = os.path.basename(input_dir) + "_" + img_name

    # 保存输出图像
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img)
    print(f"已保存: {output_path}")
