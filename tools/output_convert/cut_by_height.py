from PIL import Image, ImageEnhance
import os


def get_file_name_without_extension(image_path):
    filename_with_extension = os.path.basename(image_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension


def segment_image(image_path, n, output_path):
    # Open the image
    img = Image.open(image_path)
    width, height = img.size

    # Calculate the height of each segment
    segment_height = height // n

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Enhance color for green and red segments
    green_enhancer = ImageEnhance.Color(Image.new("RGB", (1, 1), "#00FF00"))
    red_enhancer = ImageEnhance.Color(Image.new("RGB", (1, 1), "#FF0000"))

    # Iterate over each segment and crop the image
    for i in range(n):
        top = i * segment_height
        bottom = (i + 1) * segment_height
        segment = img.crop((0, top, width, bottom))

        # # Set transparency to 50%
        # segment = segment.convert("RGBA")
        # segment_data = segment.getdata()
        # new_segment_data = [(r, g, b, a) for r, g, b, a in segment_data]
        # segment.putdata(new_segment_data)

        # # Resize the enhanced color image to match the dimensions of the segmented image
        # color_image = green_enhancer.enhance(0.5) if i == 0 else red_enhancer.enhance(0.5)
        # color_image = color_image.resize(segment.size)
        #
        # # Blend the segmented image with the enhanced color image
        # segment = Image.blend(segment, color_image, 0.5)

        filename_without_extension = get_file_name_without_extension(image_path)

        # Save segmented image
        segment.save(os.path.join(output_path, filename_without_extension + f"_segment_{i + 1}.png"), "PNG", quality=95)


# Example usage:
image_path = r"C:\Users\WIN\Desktop\4\quadrant1.png"  # Replace with your image path
output_path = r"C:\Users\WIN\Desktop\4"  # Specify output directory path
n = 2  # Number of segments

segment_image(image_path, n, output_path)
