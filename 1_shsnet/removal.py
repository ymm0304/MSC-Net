import cv2
import numpy as np


def dilate_image(I, B):
    # Perform the dilation operation
    return cv2.dilate(I, B)


def find_nearest_non_highlight_pixel(I, highlight_coords):
    # Get coordinates of non-highlight pixels
    non_highlight_coords = np.argwhere(I == 0)
    nearest_pixels = []

    for hc in highlight_coords:
        distances = np.linalg.norm(non_highlight_coords - hc, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_pixels.append(non_highlight_coords[nearest_idx])

    return np.array(nearest_pixels)


def highlight_removal(I, B):
    # Step 1: Dilate the image
    dilated_image = dilate_image(I, B)

    # Step 2: Find nearest non-highlight pixels for each highlight pixel
    highlight_coords = np.argwhere(dilated_image == 1)
    nearest_pixels = find_nearest_non_highlight_pixel(I, highlight_coords)

    # Step 3: Replace highlight pixels with the nearest non-highlight pixel values
    for i, hc in enumerate(highlight_coords):
        I[tuple(hc)] = I[tuple(nearest_pixels[i])]

    return I


def smooth_highlight_regions(I, color_image, window_size, tau):
    # Smooth highlight regions based on nearest non-specular pixel
    highlight_coords = np.argwhere(I == 1)
    result_image = color_image.copy()

    for hc in highlight_coords:
        x, y = hc
        # Define the window
        x_min = max(x - window_size // 2, 0)
        x_max = min(x + window_size // 2, color_image.shape[0] - 1)
        y_min = max(y - window_size // 2, 0)
        y_max = min(y + window_size // 2, color_image.shape[1] - 1)

        window = color_image[x_min:x_max + 1, y_min:y_max + 1]
        center_pixel = color_image[x, y]

        # Compute color similarity
        distances = np.linalg.norm(window - center_pixel, axis=2)
        mask = distances < tau

        # Compute the average color value of non-specular pixels
        non_specular_pixels = window[mask]
        if len(non_specular_pixels) > 0:
            avg_color = np.mean(non_specular_pixels, axis=0)
            result_image[x, y] = avg_color

    return result_image


# Load the binary image and the original color image
binary_image = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread('color_image.png')

# Define the structuring element for dilation
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Perform highlight removal
highlight_removed_image = highlight_removal(binary_image, structuring_element)

# Define the parameters for smoothing
window_size = 5
tau = 30

# Smooth the highlight regions
final_image = smooth_highlight_regions(highlight_removed_image, color_image, window_size, tau)

# Save the final image
cv2.imwrite('final_image.png', final_image)
