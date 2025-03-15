import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
from scipy.ndimage import label
from PIL import Image
from scipy import stats


PIXEL_THRESHOLD = 0.25


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.encoder1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = self.double_conv(512, 1024)

        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.double_conv(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.double_conv(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.double_conv(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=(1, 1))

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


def assemble_predictions_with_overlapping(windows, image_shape, window_size, stride):
    """
    Stitches the image back together from sliding windows.
    Averages overlapping windows if the stride is smaller than the window size.

    :param windows: List of sliding windows (predictions)
    :param image_shape: Tuple (height, width) of the original image shape
    :param window_size: Tuple (height, width) of the sliding window size
    :param stride: Stride of the sliding window
    :return: The stitched together image with averaged overlapping areas
    """
    image_height, image_width = image_shape[:2]
    window_height, window_width = window_size[:2]

    # Number of rows and columns of windows
    num_rows = (image_height - window_height) // stride + 1
    num_cols = (image_width - window_width) // stride + 1

    # Initialize arrays for the reconstructed image and the count of overlapping windows
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    overlap_count = np.zeros((image_height, image_width), dtype=np.float32)

    # Place each window into the correct position in the reconstructed image
    window_index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            y = row * stride
            x = col * stride
            # Upper bound by window size (should not be problem due to resizing)
            if y + window_height > image_height:
                y = image_height - window_height
            if x + window_width > image_width:
                x = image_width - window_width

            current_window = windows[window_index]

            # Check and remove singleton dimension or extra channel dimension
            if current_window.ndim == 3 and current_window.shape[0] == 1:
                current_window = np.squeeze(current_window, axis=0)

            # Ensure the window shape matches the reconstructed image slice
            if current_window.shape != (window_height, window_width):
                raise ValueError(f"Shape mismatch between window and reconstructed image slice: "
                                 f"{current_window.shape} vs {(window_height, window_width)}")

            # Ensure the window shape matches the reconstructed image slice
            if current_window.shape != (window_height, window_width):
                raise ValueError(f"Shape mismatch between window and reconstructed image slice: "
                                 f"{current_window.shape} vs {(window_height, window_width)}")

            # Add window values to the reconstructed image and increment overlap count
            reconstructed_image[y:y + window_height, x:x + window_width] += current_window
            overlap_count[y:y + window_height, x:x + window_width] += 1
            window_index += 1

    # Avoid division by zero
    overlap_count[overlap_count == 0] = 1

    # Average the overlapping regions
    reconstructed_image /= overlap_count

    return reconstructed_image


def assemble_predictions_with_overlapping_max(windows, image_shape, window_size, stride):
    """
    Stitches the image back together from sliding windows.
    Takes the maximum value from overlapping windows if the stride is smaller than the window size.

    :param windows: List of sliding windows (predictions)
    :param image_shape: Tuple (width, height) of the original image shape
    :param window_size: Tuple (width, height) of the sliding window size
    :param stride: Stride of the sliding window
    :return: The stitched together image with maximum overlapping areas
    """
    image_width, image_height = image_shape
    window_height, window_width = window_size

    # Number of rows and columns of windows
    num_rows = (image_height - window_height) // stride + 1
    num_cols = (image_width - window_width) // stride + 1

    # Initialize arrays for the reconstructed image and the maximum values
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float32)
    count_matrix = np.zeros((image_height, image_width), dtype=np.float32)  # To track the number of windows covering each pixel

    # Place each window into the correct position in the reconstructed image
    window_index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            y = row * stride
            x = col * stride
            # Upper bound by window size
            if y + window_height > image_height:
                y = image_height - window_height
            if x + window_width > image_width:
                x = image_width - window_width

            current_window = windows[window_index]
            if current_window.ndim == 3 and current_window.shape[0] == 1:
                current_window = np.squeeze(current_window, axis=0)

            # Ensure the window shape matches the reconstructed image slice
            if current_window.shape != (window_height, window_width):
                raise ValueError(f"Shape mismatch between window and reconstructed image slice: "
                                 f"{current_window.shape} vs {(window_height, window_width)}")

            # Update the reconstructed image with the maximum value from overlapping windows
            reconstructed_image[y:y + window_height, x:x + window_width] = np.maximum(
                reconstructed_image[y:y + window_height, x:x + window_width], current_window
            )
            # count_matrix[y:y + window_height, x:x + window_width] += 1

            window_index += 1

    # Normalize by the count matrix to average overlapping windows
    # reconstructed_image /= count_matrix
    reconstructed_image = np.clip(reconstructed_image, 0, 1)  # Ensure pixel values are between 0 and 1

    return reconstructed_image

def sliding_window_preloaded(image, window_size, stride):
    # Make sure stride is not 0
    if stride == 0:
        stride = 1

    image_height, image_width = image.shape[:2]
    window_height, window_width = window_size

    windows = []
    for y in range(0, image_height - window_height + 1, stride):
        for x in range(0, image_width - window_width + 1, stride):
            windows.append(image[y:y + window_height, x:x + window_width])


    return windows


def sliding_window_preloaded_padding(image, window_size, stride):
    # Make sure stride is not 0
    if stride <= 0:
        stride = 1

    image_height, image_width = image.shape[:2]
    window_height, window_width = window_size

    # Pad image to fit
    pad_bottom = (stride - (image_height % stride)) % stride
    pad_right = (stride - (image_width % stride)) % stride

    padded_image = np.pad(image, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
    padded_height, padded_width = padded_image.shape[:2]

    windows = []
    for y in range(0, padded_height - window_height + 1, stride):
        for x in range(0, padded_width - window_width + 1, stride):
            windows.append(padded_image[y:y + window_height, x:x + window_width])

    return windows, (padded_width, padded_height)


def resize_image(image, window_size, stride, div_mult=2.25, max_div_mult=2.0):
    """
    Will scale the image to the window_size + stride.
    """
    if stride == 0:
        stride = 1

    # Get image dimensions
    image_height, image_width = image.shape[:2]
    window_height, window_width = window_size

    # Calculate the scale factor for resizing
    scale_factor = 1

    # If either dimension exceeds the div_mult times the window size, resize
    if image_height > window_height * div_mult or image_width > window_width * div_mult:
        scale_factor = min(
            (window_height * max_div_mult) / image_height,  # Max height scale
            (window_width * max_div_mult) / image_width     # Max width scale
        )

    # Calculate new dimensions while maintaining the aspect ratio
    new_height = int(image_height * scale_factor)
    new_width = int(image_width * scale_factor)

    # Ensure that both dimensions are multiples of window_size + stride for the sliding window
    # new_height = calculate_resize_dimension(new_height, window_height, stride)
    # new_width = calculate_resize_dimension(new_width, window_width, stride)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate the resizing ratio for sphericity calculation
    original_area = image_height * image_width
    resized_area = new_height * new_width
    ratio = original_area / resized_area

    # Return the resized image and associated metadata
    return resized_image, (new_width, new_height), ratio


def calculate_resize_dimension(original_size, window_size, stride):
    current_size = window_size
    while current_size < original_size:
        current_size += stride

    return current_size


def load_image(image_path):
    img = cv2.imread(image_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray


def save_image(image, image_path):
    cv2.imwrite(image_path, image)


def prepare_images(images):
    """
    Prepare images and labels by normalizing them to [0, 1] and converting them to NumPy arrays.

    :param images: List of image arrays.
    :param labels: List of label arrays.
    :return: Tuple of (normalized_images, normalized_labels) where both are NumPy arrays.
    """
    # Normalize images and labels to range [0, 1]
    normalized_images = [img.astype(np.float32) / 255.0 for img in images]

    # Convert lists to NumPy arrays
    normalized_images = np.array(normalized_images)

    # Change to tensors
    normalized_images = torch.tensor(normalized_images).unsqueeze(1) # Need extra channel when grayscale for pytorch unet

    return normalized_images



def threshold_images(images, threshold=0.5):
    """
    Performs a threshold over the list of images.
    :param images: As used in the pipeline, this is the list of the assembled predictions.
    :param threshold: Threshold to create a binary mask using.
    :return: List of binary masks
    """
    binary_masks = []

    for image in images:
        binary_mask = (image > threshold).astype(np.uint8)
        binary_masks.append(binary_mask)

    return binary_masks


def threshold_image(image, threshold=0.5):
    """
    Performs a threshold over the list of images.
    :param images: As used in the pipeline, this is the list of the assembled predictions.
    :param threshold: Threshold to create a binary mask using.
    :return: List of binary masks
    """
    return (image > threshold).astype(np.uint8)


def remove_cells_at_borders(images):
    """
    Removes the cells at the borders of images, done so since this was required in the original labelling of
    the dataset.
    :param images: list of images (numpy arrays)
    :return: list of numpy arrays
    """
    processed_images = []

    for image in images:
        # create a mask to identify border touching cells
        border_mask = np.zeros_like(image, dtype=bool)

        # mark border areas as True
        border_mask[:2, :] = True # Setting to two to maybe have it work
        border_mask[-2:, :] = True
        border_mask[:, :2] = True
        border_mask[:, -2:] = True

        # label connected components
        labeled_image, num_features = label(image)

        # identify labels touching the border
        border_labels = np.unique(labeled_image[border_mask])

        # make a new image and remove all cells touching border of image
        cleaned_image = np.copy(image)
        for lbl in border_labels:
            if lbl > 0:  # ignore label 0, is background
                cleaned_image[labeled_image == lbl] = 0

        processed_images.append(cleaned_image)

    return processed_images


def remove_cells_at_borders_single_image(image):
    """
    Removes cells at border of single image
    :param image:
    :return:
    """
    # create a mask to identify border touching cells
    border_mask = np.zeros_like(image, dtype=bool)

    # mark border areas as True
    border_mask[:2, :] = True  # Setting to two to maybe have it work
    border_mask[-2:, :] = True
    border_mask[:, :2] = True
    border_mask[:, -2:] = True

    # label connected components
    labeled_image, num_features = label(image)

    # identify labels touching the border
    border_labels = np.unique(labeled_image[border_mask])

    # make a new image and remove all cells touching border of image
    cleaned_image = np.copy(image)
    for lbl in border_labels:
        if lbl > 0:  # ignore label 0, is background
            cleaned_image[labeled_image == lbl] = 0

    return cleaned_image


def sphericity_calculation(area, perimeter):
    """
    For spheres, we use (6piA^2)/(V^2). Since these are two-dimensional, we simplify this to (4pi*A) / (P^2)
    :param area: float, area of cell
    :param perimeter: float, perimeter of cell
    :return: float, sphericity of cell
    """
    return 4 * np.pi * area / (perimeter ** 2)


def find_cells_in_image(mask, ratio, sphericity_threshold=0.5, cell_size_threshold=1):
    """
    This will find the cells in an image and return them within a list
    :param mask: np array or PIL image
    :return: np array or PIL image
    """

    if isinstance(mask, Image.Image):
        mask = np.array(mask)

        # Convert mask to binary (0 or 255)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the filtered contours
    mask_ = np.zeros_like(mask)

    cell_sizes = []

    for contour in contours:
        area = cv2.contourArea(contour)  # Area of the contour
        perimeter = cv2.arcLength(contour, True)  # Perimeter of the contour

        # Avoid division by zero
        if perimeter == 0:
            continue

        # Calculate sphericity
        sphericity = (4 * np.pi * area) / (perimeter ** 2)

        # Check if the sphericity is above the threshold
        if sphericity > sphericity_threshold and area > cell_size_threshold:
            # print(sphericity)
            cv2.drawContours(mask_, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            cell_sizes.append(area * ratio)

    # Combine the mask with the original image (optional)
    return mask_, cell_sizes


def sphericity_threshold_images(image_list, ratios, threshold=0.65, cell_size_threshold=1):
    """
    Performs sphericity threshold over images
    :param image_list: list of numpy arrays
    :param threshold: Float, sphericity threshold
    :return: list of numpy arrays
    """
    thresholded_images = []
    cell_sizes = []
    for idx, image in enumerate(image_list):
        ratio = int(ratios[idx])
        # in same order as were saved (should be)
        cells, areas = find_cells_in_image(image, ratio, sphericity_threshold=threshold, cell_size_threshold=cell_size_threshold)
        thresholded_images.append(cells)
        cell_sizes.extend(areas)

    return thresholded_images, cell_sizes


def sphericity_threshold_image(image, ratio, threshold=0.65, cell_size_threshold=50):
    """
    Performs sphericity threshold over images
    :param image_list: list of numpy arrays
    :param threshold: Float, sphericity threshold
    :return: list of numpy arrays
    """
    thresholded_image, cell_sizes = find_cells_in_image(image, ratio, sphericity_threshold=threshold, cell_size_threshold=cell_size_threshold)

    return thresholded_image, cell_sizes
