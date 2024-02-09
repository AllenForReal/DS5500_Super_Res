import cv2
import numpy as np
import urllib.request
import random
import matplotlib.pyplot as plt
import os


def extract_patch(image, top_left, patch_size):
    start_h, start_w = top_left
    patch_height, patch_width = patch_size
    end_h = start_h + patch_height
    end_w = start_w + patch_width
    patch = image[start_h:end_h, start_w:end_w, :]
    return patch


def calculate_dark_percentage(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(image_gray < 10)
    total_pixels = image_gray.size
    dark_percentage = (black_pixels / total_pixels) * 100
    return dark_percentage


def rotate_image(image, angle):
    image_height, image_width, _ = image.shape
    center = (image_width // 2, image_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height))
    return rotated_image


# Load the large image
# image_url = "testImage1.png"
# resp = urllib.request.urlopen(image_url)
# image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
# image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
image_path = "testImage1.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Define the patch size, number of samples, and stride
patch_size = (650, 650)
num_samples = 200
rotation_step = 5
#
stride = max(patch_size[0] // 10, patch_size[1] // 10)

# Initialize an empty list to store all sampled patches
all_sampled_patches = []

# Loop through rotation angles from 0 to 360 degrees with a step size of 10 degrees
for rotation_angle in range(0, 361, rotation_step):
    # Rotate the image
    rotated_image = rotate_image(image, rotation_angle)

    # Initialize an empty list to store the sampled patches for the current rotation angle
    sampled_patches = []

    # Calculate the number of possible patches
    image_height, image_width, _ = rotated_image.shape
    num_patches_height = (image_height - patch_size[0]) // stride + 1
    num_patches_width = (image_width - patch_size[1]) // stride + 1
    total_num_patches = num_patches_height * num_patches_width

    # Randomly select patch indices until we have the desired number of samples
    while len(sampled_patches) < num_samples:
        patch_index = random.randint(0, total_num_patches - 1)
        patch_row = patch_index // num_patches_width
        patch_col = patch_index % num_patches_width
        top_left = (patch_row * stride, patch_col * stride)

        # Extract the patch
        patch = extract_patch(rotated_image, top_left, patch_size)

        # Check the dark zone percentage of the patch
        dark_percentage = calculate_dark_percentage(patch)

        # If the dark zone occupies more than 5% of the patch, skip this patch
        if dark_percentage > 5:
            continue

        # Add the patch to the sampled patches list
        sampled_patches.append(patch)

    # Add the sampled patches for the current rotation angle to the list of all sampled patches
    all_sampled_patches.extend(sampled_patches)

#
# Display the number of patches and the size of each patch
num_patches = len(all_sampled_patches)
patch_height, patch_width, _ = all_sampled_patches[0].shape
print("Number of sampled patches:", num_patches)
print("Patch size:", patch_height, "x", patch_width)

# Create the sample_image folder if it doesn't exist
output_folder = "datasets"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save each sampled patch as a separate image (optional)
for i, patch in enumerate(all_sampled_patches):
    patch_filename = os.path.join(output_folder, "{:05d}.a.jpg".format(i))
    cv2.imwrite(patch_filename, patch)
    print("Saved patch:", patch_filename)

# Load the first 10 patches from the data source
num_patches_to_load = 10
tuple_patches = [tuple(patch.ravel()) for patch in all_sampled_patches[:num_patches_to_load]]
original_shape = all_sampled_patches[0].shape

# Convert the tuples back to numpy arrays
loaded_patches = [np.array(tuple_patch).reshape(original_shape) for tuple_patch in tuple_patches]

# Display the loaded patches
num_rows = 2
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
axes = axes.ravel()

for i, patch in enumerate(loaded_patches):
    axes[i].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    axes[i].axis('off')

plt.tight_layout()
plt.show()

image1=cv2.imread("datasets/14599.a.jpg")
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
print(image1.shape)
print(gray_image1.shape)
