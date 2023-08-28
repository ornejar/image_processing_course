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

# Function to apply a sharpen, smooth, or Laplacian filter to an input image with specified padding
def my_imfilter(s, filter_type, pad_type='valid'):
    # Define the filters for sharpen, smooth, and Laplacian
    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    smooth_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    laplacian_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    # Choose the appropriate filter based on filter_type
    if filter_type == 'sharpen':
        filt = sharpen_filter
    elif filter_type == 'smooth':
        filt = smooth_filter
    elif filter_type == 'laplacian':
        filt = laplacian_filter
    else:
        raise ValueError("Invalid filter type. Supported types are 'sharpen', 'smooth', and 'laplacian'.")
    
    # Get the size of the input image and the filter
    h, w = s.shape
    fh, fw = filt.shape
    
    # Check if the filter is odd-sized (required for correct centering during convolution)
    if fh % 2 == 0 or fw % 2 == 0:
        raise ValueError("The filter dimensions must be odd-sized.")
    
    # Calculate padding size (use fh // 2 for both dimensions)
    pad_size_h = fh // 2
    pad_size_w = fw // 2
    
    # Apply padding to the input image based on the specified padding type
    if pad_type == 'zero':
        s_padded = np.pad(s, ((pad_size_h, pad_size_h), (pad_size_w, pad_size_w)), 'constant', constant_values=0)
    elif pad_type == 'replicate':
        s_padded = np.pad(s, ((pad_size_h, pad_size_h), (pad_size_w, pad_size_w)), 'edge')
    elif pad_type == 'reflect':
        s_padded = np.pad(s, ((pad_size_h, pad_size_h), (pad_size_w, pad_size_w)), 'reflect')
    else:
        # If pad_type is 'valid', perform convolution without padding
        s_padded = s
    
    # Initialize the result array
    result = np.zeros_like(s)
    
    # Apply convolution with padding
    for i in range(h):
        for j in range(w):
            # Extract the region of interest from the padded image
            roi = s_padded[i:i+fh, j:j+fw]
            # Perform element-wise multiplication between the filter and the ROI
            filtered_roi = roi * filt
            # Sum the elements to get the convolution result
            result[i, j] = np.sum(filtered_roi)

    
    return result

# Example usage:
# Replace 'input_image.jpg' with the path to your image
image_path = 'moon.jpg'
original_image = load_image(image_path)

# Check the shape of the original image array
print(original_image.shape)

filter_type = 'smooth'  # Choose 'sharpen', 'smooth', or 'laplacian'
pad_type = 'replicate'  # Choose 'zero', 'replicate', or 'reflect'

# Apply the filter using the my_imfilter function
filtered_image = my_imfilter(original_image, filter_type, pad_type)

# Normalize pixel values for display
filtered_image_normalized = ((filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())) * 255

# Convert the normalized image to integer values in the [0, 255] range
filtered_image_normalized = filtered_image_normalized.astype(np.uint8)

# Show the original and filtered images side by side
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image_normalized, cmap='gray')
plt.title('Filtered Image')

plt.show()