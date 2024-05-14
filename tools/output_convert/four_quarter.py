from PIL import Image
import os

# Set the paths
original_image_path = r"C:\Users\WIN\Desktop\area10_0_0_512_512.png"
output_folder = r"C:\Users\WIN\Desktop\4"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the original image
image = Image.open(original_image_path)

# Get the width and height of the image
width, height = image.size

# Calculate the dimensions of each quadrant
quad_width = width // 2
quad_height = height // 2

# Define the coordinates of the quadrants
box1 = (0, 0, quad_width, quad_height)
box2 = (quad_width, 0, width, quad_height)
box3 = (0, quad_height, quad_width, height)
box4 = (quad_width, quad_height, width, height)

# Crop and save each quadrant
quadrant1 = image.crop(box1)
quadrant1.save(os.path.join(output_folder, "quadrant1.png"))

quadrant2 = image.crop(box2)
quadrant2.save(os.path.join(output_folder, "quadrant2.png"))

quadrant3 = image.crop(box3)
quadrant3.save(os.path.join(output_folder, "quadrant3.png"))

quadrant4 = image.crop(box4)
quadrant4.save(os.path.join(output_folder, "quadrant4.png"))

print("Image division completed successfully.")
