from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Function to load the image without resizing and converting to grayscale
def load_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image = image.resize(size=(256,256))
    image_array = np.array(image)
    return image_array

def my_compose(s1, s2, m=3):
    # Ensure that m is within a valid range (0 to 8)
    if m < 0 or m > 8:
        raise ValueError("m must be between 0 and 8")
    
    # Calculate the number of bits to keep from s1
    bits_to_keep = 8 - m
    
    # Create a mask to extract the m highest bits from s2
    mask = 255 >> bits_to_keep
    
    # Extract the m highest bits from s2
    s2_m_bits = s2 >> bits_to_keep
    
    # Create the combined image
    combined_image = (s1 & mask) | (s2_m_bits << bits_to_keep)
    
    return combined_image

# Example usage:
image1_path = 's1.jpg'
image2_path = 's2.jpg'
original_s1 = load_image(image1_path)
original_s2 = load_image(image2_path)

# Check the shape of the original image array
print(original_s1.shape)
print(original_s2.shape)

m_value = 3  # Change this to your desired m value

combined_image = my_compose(original_s1, original_s2, m_value)

# Show the original and filtered images side by side
plt.subplot(1, 3, 1)
plt.imshow(original_s1, cmap='gray')
plt.title('Original s1')

plt.subplot(1, 3, 2)
plt.imshow(original_s2, cmap='gray')
plt.title('Original s2')

plt.subplot(1, 3, 3)
plt.imshow(combined_image, cmap='gray')
plt.title('combined')
plt.show()
def are_images_identical(image1, image2):
    """
    Compare two images pixel by pixel and check if they are identical.

    Args:
    image1 (numpy.ndarray): First image as a NumPy array.
    image2 (numpy.ndarray): Second image as a NumPy array.

    Returns:
    bool: True if the images are identical, False otherwise.
    """
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        return False

    # Compare each pixel of the two images
    comparison = np.all(image1 == image2)

    return comparison

# Example usage:
image_path1 = 's1.jpg'
image_path2 = 's2.jpg'

image1 = np.array(Image.open(image_path1))
image2 = np.array(Image.open(image_path2))

identical = are_images_identical(image1, image2)

if identical:
    print("The images are identical (possibly fake).")
else:
    print("The images are not identical.")
