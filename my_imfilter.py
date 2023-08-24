from PIL import Image
import cv2
import numpy as np

# Resize the image to 256X256
def load_resize_image(image_path, target_size=(256, 256)):
    # Load the image using Pillow
    image = Image.open(image_path)
    
    # Resize the image to the target size (256x256)
    image = image.resize(target_size, Image.BILINEAR)
    
    return image
# Convert the image to grayscale
def convert_to_grayscale(image):
    # Convert the image to grayscale using Pillow
    grayscale_image = image.convert("L")
    
    # Alternatively, you can use opencv-python for conversion as well:
    # grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    return grayscale_image

# def my_imfilter(s, filt, pad_type='valid'):
#     # Convert the filter to a numpy array
#     filt = np.array(filt)
    
#     # Get the size of the input image and the filter
#     h, w = s.shape
#     fh, fw = filt.shape
    
#     # Check if the filter is odd-sized (required for correct centering during convolution)
#     if fh % 2 == 0 or fw % 2 == 0:
#         raise ValueError("The filter dimensions must be odd-sized.")
    
#     # Define the padding size based on the filter dimensions
#     pad_size_h = fh // 2
#     pad_size_w = fw // 2
    
#     # Pad the input image according to the specified padding type
#     if pad_type == 'zero':
#         s_padded = np.pad(s, ((pad_size_h, pad_size_h), (pad_size_w, pad_size_w)), 'constant', constant_values=0)
#     elif pad_type == 'mirror':
#         s_padded = np.pad(s, ((pad_size_h, pad_size_h), (pad_size_w, pad_size_w)), 'reflect')
#     else:
#         # If pad_type is 'valid', perform convolution without padding
#         s_padded = s
    
#     # Initialize the output result
#     result = np.zeros_like(s)
    
#     # Perform convolution
#     for i in range(h):
#         for j in range(w):
#             # Extract the region of interest from the padded image
#             roi = s_padded[i:i+fh, j:j+fw]
#             # Perform element-wise multiplication between the filter and the ROI
#             filtered_roi = roi * filt
#             # Sum the elements to get the convolution result
#             result[i, j] = np.sum(filtered_roi)
    
#     return result

def my_imfilter(s, filt, pad_type='valid'):
    # Convert the filter to a numpy array
    filt = np.array(filt)
    
    # Get the size of the input image and the filter
    w, h = s.size  # Get the width and height
    fh, fw = filt.shape
    
    # Check if the filter is odd-sized (required for correct centering during convolution)
    if fh % 2 == 0 or fw % 2 == 0:
        raise ValueError("The filter dimensions must be odd-sized.")
    
    # Convert the Image to a NumPy array
    s_array = np.array(s)
    
    # Pad the input image according to the specified padding type
    if pad_type == 'zero':
        s_padded = np.pad(s_array, ((fh//2, fh//2), (fw//2, fw//2)), 'constant', constant_values=0)
    elif pad_type == 'mirror':
        s_padded = np.pad(s_array, ((fh//2, fh//2), (fw//2, fw//2)), 'reflect')
    else:
        # If pad_type is 'valid', perform convolution without padding
        s_padded = s_array
    
    # Initialize the output result
    result = np.zeros_like(s_array)
    
    # Perform convolution
    for i in range(h):
        for j in range(w):
            # Extract the region of interest from the padded image
            roi = s_padded[i:i+fh, j:j+fw]
            # Perform element-wise multiplication between the filter and the ROI
            filtered_roi = roi * filt
            # Sum the elements to get the convolution result
            result[i, j] = np.sum(filtered_roi)
    
    return result


# Test the function with a simple example
if __name__ == "__main__":

    # Replace 'input_image.jpg' with the path to your real photo
    image_path = 'IMG_20230226_083009.jpg'
    
    # Load and resize the image
    image = load_resize_image(image_path)
    
    # Convert the image to grayscale
    grayscale_image = convert_to_grayscale(image)
    
    # Display the grayscale image
    # grayscale_image.show()
    
    # Example filter (3x3 matrix for smoothing)
    filt = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])
    
    # Perform convolution with valid padding (no padding)
    result = my_imfilter(grayscale_image, filt, pad_type='valid')
    print("Convolution with valid padding:\n", result)

    # Perform convolution with zero padding
    result_zero_pad = my_imfilter(grayscale_image, filt, pad_type='zero')
    print("Convolution with zero padding:\n", result_zero_pad)

    # Perform convolution with mirrored padding
    result_mirror_pad = my_imfilter(grayscale_image, filt, pad_type='mirror')
    print("Convolution with mirrored padding:\n", result_mirror_pad)

    # For other filters or image processing tasks, use appropriate filter matrices
    # and call my_imfilter with the grayscale_array and the filter.

    # Display the result (you might need to use an appropriate image viewer or library)
    # For example, if you have matplotlib installed:
    import matplotlib.pyplot as plt
    plt.imshow(result, cmap='gray')
    plt.show()
