from PIL import Image
import numpy as np

# Display the filtered image (using matplotlib)
import matplotlib.pyplot as plt

# Function to load, resize, and convert the image to grayscale
def load_resize_grayscale_image(image_path, target_size=(256, 256)):
    # Load the image using Pillow
    image = Image.open(image_path)

    # # Rotate the image based on its EXIF orientation (if available)
    # image = image.rotate(image._getexif().get(0x0112, 1), expand=True)
    
    # Resize the image to the target size (256x256)
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert the image to grayscale
    grayscale_image = image.convert("L")
    
    # Convert the grayscale image to a NumPy array and flip vertically
    grayscale_array = np.array(grayscale_image)[::-1, :]
    # grayscale_array = np.array(image)[::-1, :]
    # grayscale_array = np.array(grayscale_image)
    
    return grayscale_array

# Example usage: Replace 'input_image.jpg' with the path to your image
image_path = 'IMG_20210927_145654.jpg'
grayscale_array = load_resize_grayscale_image(image_path)

# Check the shape of the grayscale array (should be (256, 256))
print(grayscale_array.shape)


# Function to apply a sharpen or smooth filter to an input image with specified padding
def my_imfilter(s, filter_type, pad_type='valid'):
    # Define the filters for sharpen and smooth
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
        raise ValueError("Invalid filter type. Supported types are 'sharpen' and 'smooth'.")
    
    # Get the size of the input image and the filter
    h, w = s.shape
    fh, fw = filt.shape
    
    # Check if the filter is odd-sized (required for correct centering during convolution)
    if fh % 2 == 0 or fw % 2 == 0:
        raise ValueError("The filter dimensions must be odd-sized.")
    
    # Perform convolution with specified padding
    if pad_type == 'valid':
        # No padding (valid convolution)
        result = np.zeros((h - fh + 1, w - fw + 1))
    else:
        # Compute padding size
        pad_size_h = fh // 2
        pad_size_w = fw // 2
        
        # Initialize the padded result
        result = np.zeros((h, w))
        
        # Apply convolution with padding
        for i in range(h):
            for j in range(w):
                # Extract the region of interest from the input image
                roi = s[max(0, i - pad_size_h):min(h, i + pad_size_h + 1),
                        max(0, j - pad_size_w):min(w, j + pad_size_w + 1)]
                
                # Check the size of the ROI (adjust for edges)
                roi_h, roi_w = roi.shape
                
                # Apply element-wise multiplication and sum for convolution
                result[i, j] = np.sum(roi * filt[:roi_h, :roi_w])
    
    return result


# Example usage:
# Replace 'grayscale_array' with the array obtained from the previous step
filter_type = 'smooth'  # Choose 'sharpen' or 'smooth'
pad_type = 'zero'  # Choose 'valid', 'zero', or 'mirror'
# Create a simple averaging filter
averaging_filter = np.ones((3, 3)) / 9.0

filtered_image = my_imfilter(grayscale_array, filter_type, pad_type)

# Normalize pixel values for display
filtered_image_normalized = ((filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())) * 255

# Convert the normalized image to integer values in the [0, 255] range
filtered_image_normalized = filtered_image_normalized.astype(np.uint8)

# Show the original imageS
plt.subplot(1, 2, 1)
plt.imshow(grayscale_array, cmap='gray')
plt.title('Original Image')

# Show the filtered image
plt.subplot(1, 2, 2)
plt.imshow(filtered_image_normalized, cmap='gray')
plt.title('Filtered Image')

plt.show()