from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to load the image without resizing and converting to grayscale
def load_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

# Example usage: Replace 'input_image.jpg' with the path to your image
image_path = 'IMG_20210927_145654.jpg'
original_image = load_image(image_path)

# Check the shape of the original image array
print(original_image.shape)

# Function to apply a sharpen, smooth, or Laplacian filter to an input image with specified padding
def my_imfilter(s, filter_type, pad_type='valid'):
    # Define the filters for sharpen, smooth, and Laplacian
    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    smooth_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
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
    
    # Apply padding to the input image based on the specified padding type
    # if pad_type == 'zero':
    #     s_padded = np.pad(s, ((fh//2, fh//2), (fw//2, fw//2)), 'constant', constant_values=0)
    # elif pad_type == 'mirror':
    #     s_padded = np.pad(s, ((fh//2, fh//2), (fw//2, fw//2)), 'reflect')
    # else:
    #     # If pad_type is 'valid', perform convolution without padding
    #     s_padded = s
    # Apply convolution with specified padding
    # if pad_type == 'valid':
    #     # No padding (valid convolution)
    #     result = np.zeros((h - fh + 1, w - fw + 1))
    # else:
    #     # Apply padding to the input image
    #     s_padded = np.pad(s, ((pad_size_h, pad_size_h), (pad_size_w, pad_size_w)), pad_type)
    # Initialize the result array
    result = np.zeros_like(s)
    
    # Calculate padding size (use fh // 2 for both dimensions)
    pad_size_h = fh // 2
    pad_size_w = fw // 2
    # Apply padding to the input image based on the specified padding type
    if pad_type != 'valid':
        s_padded = np.pad(s, ((pad_size_h, pad_size_h), (pad_size_w, pad_size_w)), pad_type)
    # Apply convolution with specified padding
    for i in range(h):
        for j in range(w):
            # Calculate the starting and ending indices for the ROI
            start_i = max(0, i - pad_size_h)
            end_i = min(h, i + pad_size_h + 1)
            start_j = max(0, j - pad_size_w)
            end_j = min(w, j + pad_size_w + 1)
            
            # Extract the region of interest from the padded input image
            roi = s_padded[start_i:end_i, start_j:end_j]
            
            # Check the size of the ROI (adjust for edges)
            roi_h, roi_w = roi.shape
            
            # Apply element-wise multiplication and sum for convolution
            result[i, j] = np.sum(roi * filt[:roi_h, :roi_w])
    
    return result




# Example usage:
# Replace 'original_image' with the array obtained from the load_image function
filter_type = 'laplacian'  # Choose 'sharpen', 'smooth', or 'laplacian'
pad_type = 'valid'  # Choose 'valid', 'zero', or 'mirror'

# Create a simple averaging filter
averaging_filter = np.ones((3, 3)) / 9.0

filtered_image = my_imfilter(original_image, filter_type, pad_type)

# Normalize pixel values for display
filtered_image_normalized = ((filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())) * 255

# Convert the normalized image to integer values in the [0, 255] range
filtered_image_normalized = filtered_image_normalized.astype(np.uint8)

# Show the original imageS
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

# Show the filtered image
plt.subplot(1, 2, 2)
plt.imshow(filtered_image_normalized, cmap='gray')
plt.title('Filtered Image')

plt.show()