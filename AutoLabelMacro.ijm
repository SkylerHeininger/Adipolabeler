inputDir = getDirectory("Choose a folder");

// Create output directory
outputDir = substring(inputDir, 0, lengthOf(inputDir) - 1) + "_counted\\";
if (!File.exists(outputDir)) {
    File.makeDirectory(outputDir);
}

// Get list of image files
list = getFileList(inputDir);

for (i = 0; i < list.length; i++) {
    filePath = inputDir + list[i];

    // open(filePath);

    run("Auto Label", "arg="+ filePath);

    // Close the image
    while (nImages > 0) {
         close();
    }
}

print("Batch processing complete! Images saved to: " + outputDir);
