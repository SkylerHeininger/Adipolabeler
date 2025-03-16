
import ij.*;
import ij.ImagePlus;
import ij.io.FileSaver;
import ij.plugin.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import ij.process.ImageProcessor;
import ij.plugin.frame.RoiManager;
import ij.gui.Roi;
import ij.gui.Overlay;
import ij.plugin.filter.ParticleAnalyzer;
import ij.measure.ResultsTable;

public class Auto_Label implements PlugIn {

    @Override
    public void run(String arg) {
        String options = Macro.getOptions(); // This retrieves options passed via run()
        String filePath = null;

        if (options != null) {
            filePath = options.replace("arg=", "").trim();
            System.out.println("Filepath received: " + filePath);
        }


        // Define the name of the virtual environment and the Python script
        String envName = "auto_label";
        String pythonScript = "utils/autolabel_image.py";
        String[] requiredPackages = {"torch", "numpy", "opencv-python", "scipy"};

        // Check if Python is available and set up the virtual environment
        if (isPythonAvailable()) {
            System.out.println("Python is available.");

            // Check if the virtual environment already exists
            if (!isVirtualEnvExists(envName)) {
                // Create a virtual environment if it doesn't exist
                if (createVirtualEnv(envName)) {
                    System.out.println("Virtual environment created.");
                } else {
                    System.out.println("Failed to create virtual environment.");
                }
            } else {
                System.out.println("Virtual environment already exists. Skipping creation.");
            }

            // Ensure packages are installed
            if (installPackages(envName, requiredPackages)) {
                System.out.println("Packages successfully downloaded.");
            } else {
                System.out.println("Packages unsuccessfully downloaded.");
                return;
            }

            // Proceed with the ImageJ image processing part
            // processImageUsingPython(envName, pythonScript);
            System.out.println("arg" + filePath);
            if (filePath != null && !filePath.isEmpty()) {
                // If a filename is provided, load and process that image
                System.out.println("arg valid");

                processImageUsingPython(filePath, envName, pythonScript);
            } else {
                // If no filename is provided, process the current open image
                processImageUsingPython(null, envName, pythonScript);
            }
        } else {
            System.out.println("Python is not available.");
        }
    }

    // Method to process the image using the Python script
    private void processImageUsingPython(String filePath, String envName, String pythonScript) {
        ImagePlus imp;
        boolean save = false;
        if (filePath != null) {
            System.out.println("Loading image");
            // If a filename is provided, open the image from the path
            imp = IJ.openImage(filePath);
            if (imp == null) {
                System.out.println("Error: Image file could not be opened.");
                return;
            }
            save = true;
        } else {
            // Get the current image from ImageJ (if no filename is provided)
            imp = IJ.getImage();
        }

        // Create a duplicate of the original image to apply ROIs
        String originalTitle = imp.getTitle();
        ImagePlus original = imp.duplicate();
        if (originalTitle.toLowerCase().endsWith(".tif")) {
            imp.setTitle(originalTitle.substring(0, originalTitle.length() - 4) + "_counted.tif");
        } else {
            imp.setTitle(originalTitle + "_counted");
        }
        original.setTitle(originalTitle);

        // Save the image to a temporary file
        String tempFilePath = System.getProperty("java.io.tmpdir") + "/image_for_python.tiff";
        FileSaver fileSaver = new FileSaver(imp);
        fileSaver.saveAsTiff(tempFilePath);

        // Ensure the Python script exists before proceeding
        if (!Files.exists(Paths.get(pythonScript))) {
            System.out.println("Python script not found: " + pythonScript);
            return;  // Exit if the script isn't found
        }

        // Run the Python script using the Python virtual environment
        System.out.println("Running Python script...");
        runPythonScript(envName, pythonScript, tempFilePath);

        // Load the segmentation mask
        ImagePlus maskImage = IJ.openImage(tempFilePath);
        // maskImage.show();

        if (maskImage != null) {
            // Set threshold for image
            maskImage.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE);

            // Convert mask into ROIs using ParticleAnalyzer
            RoiManager rm = new RoiManager();
            ResultsTable rt = new ResultsTable();
            ParticleAnalyzer pa = new ParticleAnalyzer(ParticleAnalyzer.ADD_TO_MANAGER, 0, rt, 750, Double.POSITIVE_INFINITY);
            pa.analyze(maskImage);

            int totalCells = rt.getCounter();
            System.out.println("Total ROIs detected: " + totalCells);

            // Iterate and print all detected cell sizes
            // System.out.println("Cell sizes (in pixels²):");
            // for (int i = 0; i < totalCells; i++) {
            //     double area = rt.getValue("Area", i);
            //     System.out.println("Cell " + (i + 1) + ": " + area + " pixels²");
            // }

            // Create an overlay
            Overlay overlay = new Overlay();

            // Add the ROIs from RoiManager to the overlay
            for (int i = 0; i < rm.getCount(); i++) {
                Roi roi = rm.getRoi(i);
                overlay.add(roi);
            }

            // Set the overlay to the image
            imp.setOverlay(overlay);
            imp.saveRoi();


            // ROIs are now stored in RoiManager
            System.out.println("ROIs detected: " + rm.getCount());

            if (save) {
                System.out.println("Saving");
                String originalPath = imp.getOriginalFileInfo() != null ? imp.getOriginalFileInfo().directory : "";
                File originalDir = new File(originalPath);

                // Ensure the '_counted' directory exists
                File parentDir = originalDir.getParentFile();
                if (parentDir == null) {
                    System.err.println("Error: Could not determine the parent directory.");
                    return;
                }

                // Create '_counted' directory inside the parent directory
                String countedDirPath = parentDir.getAbsolutePath() + File.separator + originalDir.getName() + "_counted";
                File countedDir = new File(countedDirPath);
                if (!countedDir.exists()) {
                    countedDir.mkdirs();  // Create the directory if it doesn't exist
                }

                // Generate the save path
                String savePath = countedDirPath + File.separator + imp.getTitle();
                String roiSavePath = countedDirPath + File.separator + imp.getTitle() + "_rois.zip";

                // Save the image with the overlay
                FileSaver overlaySaver = new FileSaver(imp);
                overlaySaver.saveAsTiff(savePath);
                System.out.println("Image saved to: " + savePath);

                rm.runCommand("Save", roiSavePath);
                System.out.println("ROIs saved to: " + roiSavePath);

                rm.reset();
                rm.close();
            } else {
                original.show();
            }

        } else {
            System.out.println("Error: Processed image could not be loaded.");
        }


    }

    // Check if Python is available
    public static boolean isPythonAvailable() {
        try {
            Process process = new ProcessBuilder("python", "--version")
                    .redirectErrorStream(true)
                    .start();

            // Read the output of the command (which contains the version of Python)
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);  // This prints the Python version
            }

            int exitCode = process.waitFor();
            return exitCode == 0;  // If exit code is 0, Python is available
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return false;  // If an exception occurs, assume Python is not available
        }
    }

    // Check if the virtual environment exists
    public static boolean isVirtualEnvExists(String envName) {
        // Check if the directory for the virtual environment exists
        File envDir = new File(envName);
        return envDir.exists() && envDir.isDirectory();
    }

    // Create the virtual environment
    public static boolean createVirtualEnv(String envName) {
        try {
            // Command to create a virtual environment (uses "python -m venv <envName>")
            Process process = new ProcessBuilder("python", "-m", "venv", envName)
                    .redirectErrorStream(true)
                    .start();

            // Capture the output from the process
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);  // This prints the output of the command
            }

            int exitCode = process.waitFor();
            return exitCode == 0;  // If the exit code is 0, it was successful
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return false;  // If an exception occurs, assume creation failed
        }
    }

    // Install required packages in the virtual environment
    public static boolean installPackages(String envName, String[] packages) {
        // Command to use pip for installing packages
        // On Windows, the path would be <envName>\Scripts\pip.exe
        String pipPath = envName + "/Scripts/pip3.12.exe";

        // Loop through each package and check if it's installed
        for (String pkg : packages) {
            // Check if the package is installed by checking the pip list
            if (!isPackageInstalled(envName, pkg)) {
                try {
                    // Install the package using pip
                    Process process = new ProcessBuilder(pipPath, "install", pkg)
                            .redirectErrorStream(true)
                            .start();

                    // Capture the output from the process
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }

                    int exitCode = process.waitFor();
                    if (exitCode == 0) {
                        System.out.println(pkg + " has been installed successfully.");
                    } else {
                        System.out.println("Failed to install " + pkg);
                        return false;  // If installation fails, return false
                    }
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                    return false;  // If an exception occurs, assume installation failed
                }
            } else {
                System.out.println(pkg + " is already installed.");
            }
        }
        return true;  // All packages installed or already present
    }

    // Check if the required package is installed in the virtual environment
    public static boolean isPackageInstalled(String envName, String packageName) {
        try {
            // Determine the correct pip path based on the operating system
            String pipPath = getPipPath(envName);

            // Command to list all installed packages
            Process process = new ProcessBuilder(pipPath, "list")
                    .redirectErrorStream(true)
                    .start();

            // Capture the output from the process
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            boolean packageFound = false;

            // Loop through the pip list output
            while ((line = reader.readLine()) != null) {
                // Check if the line starts with the package name
                if (line.startsWith(packageName + " ")) {
                    packageFound = true;
                    break;  // If the package is found, stop checking
                }
            }

            int exitCode = process.waitFor();
            return exitCode == 0 && packageFound;  // If the package is found, return true
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return false;  // If an exception occurs, assume the package is not installed
        }
    }


    // Get the path to pip based on the operating system
    public static String getPipPath(String envName) {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            // System.out.println("Windows detected");
            return envName + "/Scripts/pip3.12.exe";  // On Windows, it's in the Scripts folder
        } else {
            // System.out.println("Not windows");
            return envName + "/bin/pip";  // On Linux/macOS, it's in the bin folder
        }
    }

    // Run the Python script using the virtual environment
    public static void runPythonScript(String envName, String scriptPath, String filepath) {
        try {
            // Command to run the Python script using the virtual environment's Python
            String pythonPath = envName + "/Scripts/python.exe";
            Process process = new ProcessBuilder(pythonPath, scriptPath, filepath)
                    .redirectErrorStream(true)
                    .start();

            // Capture the output from the process (to read any print statements from Python)
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Python script executed successfully.");
            } else {
                System.out.println("Python script execution failed.");
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

}
