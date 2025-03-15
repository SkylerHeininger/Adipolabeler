import sys
from PIL import Image
import numpy as np
# Utils includes: UNet, assemble_predictions_with_overlapping, make_predictions,
# sliding_window_preloaded, resize_image, calculate_resize_dimension, load_image
from utils import *
import config
import glob
from torch.utils.data import DataLoader, TensorDataset


if __name__ == "__main__":
    # Get the image path from the argument passed by Java
    image_path = sys.argv[1]
    gray_image = load_image(image_path)

    original_dimensions = gray_image.shape
    print(original_dimensions)

    window_size = config.WINDOW_SIZE
    stride = config.STRIDE
    model_name = config.MODEL_NAME
    div_ratio = config.DIV_RATIO

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1)  # Adjust as necessary
    model.load_state_dict(torch.load(str(model_name), map_location=device))
    model.to(device)
    model.eval()

    resized_image, resized_dimensions, ratio = resize_image(gray_image, window_size, stride)
    print(resized_dimensions)

    slid_image, padded_dims = sliding_window_preloaded_padding(resized_image, window_size, stride)

    tensored_slid_images = prepare_images(slid_image)

    # Convert test_images to DataLoader
    test_dataset = TensorDataset(tensored_slid_images)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images_batch = batch[0].to(device)
            outputs = model(images_batch)
            predictions.append(outputs.cpu().numpy())

    # print(f"len of predictions: {len(predictions)}")
    predictions = np.concatenate(predictions, axis=0)


    stitched_image = assemble_predictions_with_overlapping_max(predictions, padded_dims,
                                                               window_size, stride)

    stitched_image = stitched_image[:resized_dimensions[1], :resized_dimensions[0]]

    print(stitched_image.shape)

    stitched_image_reshaped = cv2.resize(stitched_image, (original_dimensions[1], original_dimensions[0]))


    img_scaled = np.clip(stitched_image_reshaped * 255, 0, 255).astype(np.uint8)

    save_image(img_scaled, "utils/test_output.png")

    # change threshold as needed
    masked_prediction = threshold_image(stitched_image_reshaped, threshold=PIXEL_THRESHOLD)
    print(masked_prediction.shape)

    processed_prediction = remove_cells_at_borders_single_image(masked_prediction)

    # Perform sphericity processing
    thresholded_image, cell_sizes = sphericity_threshold_image(processed_prediction, ratio,
                                                               threshold=config.SPHERICITY_THRESHOLD,
                                                               cell_size_threshold=config.CELL_AREA_THRESHOLD)
    print(len(cell_sizes))

    print(f"Average cell size was: {sum(cell_sizes) / len(cell_sizes)}")

    print(thresholded_image.shape)

    save_image(thresholded_image, image_path)