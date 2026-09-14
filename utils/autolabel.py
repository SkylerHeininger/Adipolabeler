import sys
import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    UNet,
    resize_image,
    sliding_window_preloaded_padding,
    prepare_images,
    assemble_predictions_with_overlapping_max,
    threshold_image,
    remove_cells_at_borders_single_image,
    sphericity_threshold_image,
    load_image,
    save_image,
)
import config
from model_loader import get_model_path


def load_model(device):
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(get_model_path(), map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(image, model, device):
    original_dimensions = image.shape

    if config.USE_SCALING:
        scale_up = 1.0 / config.SCALING
        image = cv2.resize(image, None, fx=scale_up, fy=scale_up)

    window_size = config.WINDOW_SIZE
    stride = config.STRIDE

    resized_image, resized_dimensions, ratio = resize_image(image, window_size, stride)
    slid_image, padded_dims = sliding_window_preloaded_padding(resized_image, window_size, stride)
    tensored_slid_images = prepare_images(slid_image)

    test_dataset = TensorDataset(tensored_slid_images)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images_batch = batch[0].to(device)
            outputs = model(images_batch)
            predictions.append(outputs.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)

    stitched_image = assemble_predictions_with_overlapping_max(predictions, padded_dims, window_size, stride)
    stitched_image = stitched_image[:resized_dimensions[1], :resized_dimensions[0]]

    intermediate_dims = image.shape[:2] if config.USE_SCALING else original_dimensions
    stitched_image_reshaped = cv2.resize(stitched_image, (intermediate_dims[1], intermediate_dims[0]))

    if config.USE_SCALING:
        stitched_image_reshaped = cv2.resize(
            stitched_image_reshaped, (original_dimensions[1], original_dimensions[0])
        )

    masked_prediction = threshold_image(stitched_image_reshaped, threshold=config.PIXEL_THRESHOLD)
    processed_prediction = remove_cells_at_borders_single_image(masked_prediction)

    final_ratio = ratio * (config.SCALING ** 2) if config.USE_SCALING else ratio

    thresholded_image, cell_sizes = sphericity_threshold_image(
        processed_prediction, final_ratio,
        threshold=config.SPHERICITY_THRESHOLD,
        cell_size_threshold=config.CELL_AREA_THRESHOLD,
    )

    return thresholded_image, cell_sizes


def process_single(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    gray_image = load_image(image_path)
    thresholded_image, cell_sizes = run_inference(gray_image, model, device)

    if cell_sizes:
        print(f"Average cell size was: {sum(cell_sizes) / len(cell_sizes)}")
    else:
        print("No cells detected.")

    save_image(thresholded_image, image_path)


def process_batch(input_directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    files = glob.glob(f"{input_directory}*.tif")
    os.makedirs("temp", exist_ok=True)

    for file in files:
        gray_image = load_image(file)
        thresholded_image, cell_sizes = run_inference(gray_image, model, device)
        image_path = f"temp/{os.path.basename(file)}"
        save_image(thresholded_image, image_path)

    return len(files)


if __name__ == "__main__":
    image_path = sys.argv[1]
    batched = sys.argv[2].lower() == "true"

    if batched:
        process_batch(image_path)
    else:
        process_single(image_path)