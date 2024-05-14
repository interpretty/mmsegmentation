import mmcv
import numpy as np
import os


def convert_pixel(pixel_value):
    if np.array_equal(pixel_value, [0, 0, 0]):
        return [0, 0, 0]  # Black
    elif np.array_equal(pixel_value, [1, 1, 1]):
        return [255, 255, 255]  # White
    elif np.array_equal(pixel_value, [2, 2, 2]):
        return [255, 0, 0]  # Red
    elif np.array_equal(pixel_value, [3, 3, 3]):
        return [255, 255, 0]  # Yellow
    elif np.array_equal(pixel_value, [4, 4, 4]):
        return [0, 255, 0]  # Green
    elif np.array_equal(pixel_value, [5, 5, 5]):
        return [0, 255, 255]  # Cyan
    elif np.array_equal(pixel_value, [6, 6, 6]):
        return [0, 0, 255]  # Blue
    else:
        return [0, 0, 0]  # Default to black if no match is found


def process_image(image_path, output_folder):
    # Read image using mmcv
    img = mmcv.imread(image_path)

    # Apply the conversion function to each pixel
    converted_array = np.array([[convert_pixel(pixel) for pixel in row] for row in img])

    # # Convert the resulting numpy array back to an image
    # converted_image = mmcv.imfromarray(np.uint8(converted_array))

    # Save the converted image to the output folder
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + "_converted.png")
    mmcv.imwrite(converted_array, output_path)
    print(f"Image saved at: {output_path}")


# Input and output folders
input_folder = r'C:\PY\mmsegmentation\data\potsdam\ann_dir\val'
output_folder = r'C:\PY\mmsegmentation\data\potsdam\ann_dir\test'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each PNG file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)
