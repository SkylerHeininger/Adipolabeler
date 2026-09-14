# Adipolabeler

Semi-automated adipocyte segmentation plugin for ImageJ/Fiji.

## Setup

### 1. Download the plugin

1. Download or clone this repository to your computer. (click  "<> Code" and then "Download ZIP")
2. Keep the folder structure intact, `Adipo_Label.java` and the `utils/` folder must stay next to each other.
3. Move the whole folder somewhere inside your ImageJ or Fiji installation's `plugins` folder (a subfolder is fine; the exact name and nesting don't matter).

### 2. Install Python

1. Install Python 3 if it isn't already on your computer, from python.org or your system's package manager.
2. Confirm it installed correctly by opening a terminal/command prompt and running `python3 --version` (or `python --version` on Windows). You may need to restart your computer.
3. You do not need to install any Python packages yourself, the plugin installs its own dependencies automatically on first run. Consequently, the first run will take much longer, simply to install the dependencies and the model.

### 3. Compile the plugin in ImageJ/Fiji

1. Open ImageJ or Fiji.
2. Go to `Plugins → Compile and Run...`.
3. Navigate to and select `Adipo_Label.java` inside the folder you placed in step 1.
4. ImageJ will compile the plugin and it will now appear under the `Plugins` menu as "Adipo Label".

### 4. First-time setup

1. Run the plugin once from `Plugins → Adipo Label`.
2. A folder-selection dialog will appear: Select the folder containing `Adipo_Label.java` and `utils/` (the same folder from step 1).
3. This selection is remembered automatically, so you will not be asked again on future runs on the same computer. Should you need to change the location, the
4. A second dialog will appear asking whether to enable bias-correction scaling. Leave this unchecked unless you need it.
5. The plugin will now automatically create a Python virtual environment and install its required packages; this may take a minute or two the first time only.
6. The trained model will download automatically from Hugging Face (https://huggingface.co/skyle123/Adipolabeler) the first time it's needed; no manual download is required. You do not need to be authenticated in Huggingface for this download, it is small enough that it downloads fast, even with rate limits.

### 5. Running on a single image

1. Open the image you want to process in ImageJ.
2. Go to `Plugins → Adipo Label`.
3. Choose your scaling settings if prompted, then let the plugin run.
4. The detected cells will appear as an overlay on the image, and results are saved automatically if the image was opened from a file.

### 6. Running on a folder of images (batch mode)

1. Go to `Plugins → Macros → Run...`.
2. Select `AdipoLabelMacro.ijm` from the plugin folder.
3. Choose the folder containing the images you want to process when prompted.
4. Processed images and ROIs will be saved automatically into a new folder alongside your input folder, with `_counted` added to its name.

### 7. Running batch mode with bias-correction scaling

1. Go to `Plugins → Macros → Run...`.
2. Select `AdipoLabelMacro_bias_test.ijm` instead of the standard macro.
3. Choose the folder containing the images you want to process when prompted.
4. This runs the same batch process as step 6, but with scaling enabled automatically, without showing the settings dialog.

### Troubleshooting

1. If the plugin reports that Python was not found, confirm Python 3 is installed and available on your system PATH, then restart ImageJ.
2. If package installation fails, check the ImageJ console log (`Window → Console`) for the specific pip error. Or, select `Edit → Options → Misc → Debug Mode` and re-run the plugin to see the full error.
3. If processing fails due to invalid dependencies, the plugin will automatically attempt one dependency reinstall and retry before reporting a final error — check the console log if it still fails after that.


## Future Updates