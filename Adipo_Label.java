import ij.*;
import ij.gui.GenericDialog;
import ij.io.FileSaver;
import ij.plugin.*;
import ij.process.ImageProcessor;
import ij.plugin.frame.RoiManager;
import ij.gui.Roi;
import ij.gui.Overlay;
import ij.plugin.filter.ParticleAnalyzer;
import ij.measure.ResultsTable;
import ij.measure.Calibration;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Adipo_Label implements PlugIn {

    private static final String PREFS_KEY = "adipolabeler.base_dir";

    private static String BASE_DIR;
    private static String ENV_NAME;
    private static String PYTHON_SCRIPT;
    private static String REQUIREMENTS;
    private static String REQUIREMENTS_HASH_FILE;
    private static String SETTINGS_FILE;
    private static String TEMP_DIR;

    private static boolean resolveBaseDir() {
        String saved = Prefs.get(PREFS_KEY, null);
        System.out.println("Prefs cache lookup: key=" + PREFS_KEY + " value=" + saved);
        if (saved != null) {
            System.out.println("Prefs value found. 'utils' subfolder exists: " + new File(saved, "utils").isDirectory());
        }

        if (saved != null && new File(saved, "utils").isDirectory()) {
            System.out.println("Using cached BASE_DIR, skipping folder selection: " + saved);
            BASE_DIR = saved;
            assignDerivedPaths();
            return true;
        }

        System.out.println("Prefs cache miss or invalid. Prompting for folder selection.");
        String chosen = IJ.getDirectory("Select the Adipolabeler folder (contains Adipo_Label.java and utils/)");
        if (chosen == null) {
            return false;
        }
        if (chosen.endsWith(File.separator)) {
            chosen = chosen.substring(0, chosen.length() - 1);
        }
        if (!new File(chosen, "utils").isDirectory()) {
            IJ.error("Adipolabeler", "The selected folder does not contain a 'utils' subfolder. "
                    + "Select the folder that directly contains Adipo_Label.java and utils/.");
            return false;
        }

        BASE_DIR = chosen;
        Prefs.set(PREFS_KEY, BASE_DIR);
        Prefs.savePreferences();
        System.out.println("Saved new BASE_DIR to Prefs: " + BASE_DIR);
        assignDerivedPaths();
        return true;
    }

    private static void assignDerivedPaths() {
        ENV_NAME = BASE_DIR + File.separator + "auto_label";
        PYTHON_SCRIPT = BASE_DIR + File.separator + "utils" + File.separator + "autolabel.py";
        REQUIREMENTS = BASE_DIR + File.separator + "utils" + File.separator + "requirements.txt";
        REQUIREMENTS_HASH_FILE = BASE_DIR + File.separator + "utils" + File.separator + ".requirements_hash";
        SETTINGS_FILE = BASE_DIR + File.separator + "utils" + File.separator + "user_settings.json";
        TEMP_DIR = BASE_DIR + File.separator + "temp";
    }

    private static String readConfigRawValue(String key) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(BASE_DIR, "utils", "config.py"));
            Pattern pattern = Pattern.compile("^\\s*" + Pattern.quote(key) + "\\s*=\\s*([A-Za-z0-9_.]+)");
            for (String line : lines) {
                Matcher matcher = pattern.matcher(line);
                if (matcher.find()) {
                    return matcher.group(1);
                }
            }
        } catch (Exception e) {
            System.err.println("Could not read " + key + " from config.py: " + e.getMessage());
        }
        return null;
    }

    private static double readConfigDouble(String key, double fallback) {
        String value = readConfigRawValue(key);
        if (value == null) {
            return fallback;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static boolean readConfigBool(String key, boolean fallback) {
        String value = readConfigRawValue(key);
        return value == null ? fallback : value.equalsIgnoreCase("true");
    }

    @Override
    public void run(String arg) {
        if (!resolveBaseDir()) {
            IJ.error("Adipolabeler", "Setup was cancelled or the Adipolabeler folder could not be located.");
            return;
        }

        String options = Macro.getOptions();
        Map<String, String> paramMap = new HashMap<>();
        if (options != null) {
            for (String pair : options.split("\n")) {
                String[] keyValue = pair.split("=", 2);
                if (keyValue.length == 2) {
                    paramMap.put(keyValue[0].trim(), keyValue[1].trim());
                }
            }
        }

        String filePath = paramMap.get("arg");
        boolean batching = Boolean.parseBoolean(paramMap.get("batch"));

        // Read directly from config.py every time, rather than duplicating
        // these numbers as Java literals -- config.py is the single source
        // of truth for defaults.
        final boolean DEFAULT_USE_SCALING = readConfigBool("USE_SCALING", false);
        final double DEFAULT_SCALING_FACTOR = readConfigDouble("SCALING", 0.75);
        final double DEFAULT_SPHERICITY_THRESHOLD = readConfigDouble("SPHERICITY_THRESHOLD", 0.5);
        final double DEFAULT_CELL_AREA_THRESHOLD_PIXELS = readConfigDouble("CELL_AREA_THRESHOLD", 750);

        boolean useScaling;
        double scalingFactor;
        double sphericityThreshold;
        double cellSizeThresholdPixels;
        int batchSize;
        boolean saveToFile;
        boolean resetToScriptDefaults;

        if (options != null) {
            useScaling = paramMap.containsKey("scaling")
                    ? Boolean.parseBoolean(paramMap.get("scaling"))
                    : DEFAULT_USE_SCALING;
            scalingFactor = paramMap.containsKey("scale_factor")
                    ? Double.parseDouble(paramMap.get("scale_factor"))
                    : DEFAULT_SCALING_FACTOR;
            sphericityThreshold = paramMap.containsKey("sphericity")
                    ? Double.parseDouble(paramMap.get("sphericity"))
                    : DEFAULT_SPHERICITY_THRESHOLD;
            cellSizeThresholdPixels = paramMap.containsKey("cell_size")
                    ? Double.parseDouble(paramMap.get("cell_size"))
                    : DEFAULT_CELL_AREA_THRESHOLD_PIXELS;
            batchSize = paramMap.containsKey("batch_size")
                    ? Integer.parseInt(paramMap.get("batch_size"))
                    : 0;
            saveToFile = paramMap.containsKey("save")
                    ? Boolean.parseBoolean(paramMap.get("save"))
                    : false;
            resetToScriptDefaults = paramMap.containsKey("reset")
                    ? Boolean.parseBoolean(paramMap.get("reset"))
                    : false;
        } else {
            double pixelWidth = 1.0;
            double pixelHeight = 1.0;
            String calUnit = "pixels";
            boolean calibrated = false;

            ImagePlus current = WindowManager.getCurrentImage();
            if (current != null) {
                Calibration cal = current.getCalibration();
                if (cal != null && cal.scaled()) {
                    pixelWidth = cal.pixelWidth;
                    pixelHeight = cal.pixelHeight;
                    calUnit = cal.getUnit();
                    calibrated = true;
                }
            }

            double pixelAreaFactor = pixelWidth * pixelHeight;
            double defaultCellSizeDisplay = calibrated
                    ? DEFAULT_CELL_AREA_THRESHOLD_PIXELS * pixelAreaFactor
                    : DEFAULT_CELL_AREA_THRESHOLD_PIXELS;

            GenericDialog gd = new GenericDialog("Adipolabeler Settings");
            gd.addCheckbox("Enable bias-correction scaling", DEFAULT_USE_SCALING);
            gd.addNumericField("Scaling factor", DEFAULT_SCALING_FACTOR, 2);
            gd.addNumericField("Sphericity threshold (0-1)", DEFAULT_SPHERICITY_THRESHOLD, 2);
            gd.addNumericField("Cell size threshold (" + calUnit + "\u00b2)", defaultCellSizeDisplay, 2);
            gd.addNumericField("Batch size (0 = auto-detect)", 0, 0);
            if (!calibrated) {
                gd.addMessage("No calibrated image detected; cell size threshold is in raw pixels\u00b2.");
            }
            gd.addMessage("Batch size controls how many image tiles are processed at once. "
                    + "Leave at 0 to auto-detect a safe value for this computer, or set manually to "
                    + "trade off speed against how much of the CPU/GPU is used.");
            gd.addCheckbox("Save these settings to file for future runs", false);
            gd.addCheckbox("Reset to script defaults (clear any previously saved settings)", false);
            gd.addMessage("By default, settings above apply only to this run and are not saved. "
                    + "Values always come from utils/config.py unless a saved settings file exists.");
            gd.showDialog();
            if (gd.wasCanceled()) {
                return;
            }
            useScaling = gd.getNextBoolean();
            scalingFactor = gd.getNextNumber();
            sphericityThreshold = gd.getNextNumber();
            double enteredCellSize = gd.getNextNumber();
            cellSizeThresholdPixels = calibrated ? enteredCellSize / pixelAreaFactor : enteredCellSize;
            batchSize = (int) gd.getNextNumber();
            saveToFile = gd.getNextBoolean();
            resetToScriptDefaults = gd.getNextBoolean();
        }

        if (resetToScriptDefaults) {
            new File(SETTINGS_FILE).delete();
        } else if (saveToFile) {
            writeUserSettings(useScaling, scalingFactor, sphericityThreshold, cellSizeThresholdPixels, batchSize);
        }

        String pythonCommand = resolvePythonCommand();
        if (pythonCommand == null) {
            IJ.error("Adipolabeler", "Python was not found on this system. "
                    + "Install Python 3 and ensure it is available on your PATH.");
            return;
        }

        if (!isVirtualEnvExists(ENV_NAME)) {
            if (!createVirtualEnv(pythonCommand, ENV_NAME)) {
                IJ.error("Adipolabeler", "Failed to create the Python virtual environment.");
                return;
            }
        }

        if (requirementsChanged()) {
            if (!installPackages(ENV_NAME, REQUIREMENTS)) {
                IJ.error("Adipolabeler", "Failed to install required Python packages. "
                        + "Check the console log for pip output.");
                return;
            }
            updateRequirementsHash();
        }

        if (!Files.exists(Paths.get(PYTHON_SCRIPT))) {
            IJ.error("Adipolabeler", "Python script not found: " + PYTHON_SCRIPT);
            return;
        }

        if (batching) {
            processImagesUsingPython(filePath, ENV_NAME);
        } else if (filePath != null && !filePath.isEmpty()) {
            processImageUsingPython(filePath, ENV_NAME);
        } else {
            processImageUsingPython(null, ENV_NAME);
        }

        deleteDir(new File(TEMP_DIR));
    }

    private void writeUserSettings(boolean useScaling, double scalingFactor,
                                    double sphericityThreshold, double cellSizeThresholdPixels,
                                    int batchSize) {
        String json = String.format(
                "{\"use_scaling\": %s, \"scaling\": %.4f, \"sphericity_threshold\": %.4f, "
                        + "\"cell_area_threshold\": %.4f, \"batch_size\": %d}",
                useScaling, scalingFactor, sphericityThreshold, cellSizeThresholdPixels, batchSize
        );
        try (PrintWriter out = new PrintWriter(new FileWriter(SETTINGS_FILE))) {
            out.println(json);
        } catch (IOException e) {
            System.err.println("Failed to write settings: " + e.getMessage());
        }
    }

    private void processImageUsingPython(String filePath, String envName) {
        ImagePlus imp;
        boolean save = false;
        if (filePath != null) {
            imp = IJ.openImage(filePath);
            if (imp == null) {
                IJ.error("Adipolabeler", "Image file could not be opened: " + filePath);
                return;
            }
            save = true;
        } else {
            imp = IJ.getImage();
        }

        String originalTitle = imp.getTitle();
        ImagePlus original = imp.duplicate();
        if (originalTitle.toLowerCase().endsWith(".tif")) {
            imp.setTitle(originalTitle.substring(0, originalTitle.length() - 4) + "_counted");
        } else {
            imp.setTitle(originalTitle + "_counted");
        }
        original.setTitle(originalTitle);

        String tempFilePath = System.getProperty("java.io.tmpdir") + "/image_for_python.tiff";
        new FileSaver(imp).saveAsTiff(tempFilePath);

        if (!runPythonScript(envName, PYTHON_SCRIPT, tempFilePath, "False")) {
            return;
        }

        ImagePlus maskImage = IJ.openImage(tempFilePath);
        if (maskImage == null) {
            IJ.error("Adipolabeler", "Processed image could not be loaded.");
            return;
        }

        maskImage.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE);

        RoiManager rm = new RoiManager();
        ResultsTable rt = new ResultsTable();
        ParticleAnalyzer pa = new ParticleAnalyzer(ParticleAnalyzer.ADD_TO_MANAGER, 0, rt, 1, Double.POSITIVE_INFINITY);
        pa.analyze(maskImage);

        Overlay overlay = new Overlay();
        for (int i = 0; i < rm.getCount(); i++) {
            overlay.add(rm.getRoi(i));
        }
        imp.setOverlay(overlay);
        imp.saveRoi();

        if (save) {
            String originalPath = imp.getOriginalFileInfo() != null ? imp.getOriginalFileInfo().directory : "";
            File originalDir = new File(originalPath);
            File parentDir = originalDir.getParentFile();
            if (parentDir == null) {
                IJ.error("Adipolabeler", "Could not determine the parent directory for saving.");
                return;
            }

            String countedDirPath = parentDir.getAbsolutePath() + File.separator + originalDir.getName() + "_counted";
            new File(countedDirPath).mkdirs();

            String roisDirPath = countedDirPath + File.separator + "roi_images";
            new File(roisDirPath).mkdirs();

            String savePath = countedDirPath + File.separator + imp.getTitle() + ".tif";
            String roiImageSavePath = roisDirPath + File.separator + imp.getTitle() + ".tif";
            String roiSavePath = countedDirPath + File.separator + imp.getTitle() + "_rois.zip";

            new FileSaver(imp).saveAsTiff(savePath);
            new FileSaver(maskImage).saveAsTiff(roiImageSavePath);

            if (rm.getCount() > 0) {
                rm.runCommand("Save", roiSavePath);
            }

            rm.reset();
            rm.close();
        } else {
            original.show();
        }
    }

    private int processImagesUsingPython(String inputDirectory, String envName) {
        if (!runPythonScript(envName, PYTHON_SCRIPT, inputDirectory, "True")) {
            return -1;
        }

        File directory = new File(TEMP_DIR);
        if (!directory.exists() || !directory.isDirectory()) {
            IJ.error("Adipolabeler", "Expected output directory 'temp' was not found.");
            return -1;
        }

        File[] files = directory.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".tif") || name.toLowerCase().endsWith(".tiff")
                        || name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg"));

        if (files == null || files.length == 0) {
            IJ.log("No image files found in temp");
            return -1;
        }

        int count = files.length;

        for (File file : files) {
            ImagePlus maskImage = IJ.openImage(file.getAbsolutePath());
            if (maskImage == null) {
                System.err.println("Could not open image " + file.getName());
                continue;
            }

            maskImage.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE);

            RoiManager rm = new RoiManager();
            ResultsTable rt = new ResultsTable();
            ParticleAnalyzer pa = new ParticleAnalyzer(ParticleAnalyzer.ADD_TO_MANAGER, 0, rt, 1, Double.POSITIVE_INFINITY);
            pa.analyze(maskImage);

            Overlay overlay = new Overlay();
            for (int i = 0; i < rm.getCount(); i++) {
                overlay.add(rm.getRoi(i));
            }

            String targetName = file.getName();
            File searchDir = new File(inputDirectory);
            File[] matchingFiles = searchDir.listFiles((dir, name) -> name.endsWith(targetName));

            if (matchingFiles == null || matchingFiles.length == 0) {
                System.err.println("No matching source file found for: " + targetName);
                continue;
            }
            File matchedFile = matchingFiles[0];

            ImagePlus imp = IJ.openImage(matchedFile.getAbsolutePath());
            imp.setOverlay(overlay);
            imp.saveRoi();

            String originalPath = imp.getOriginalFileInfo() != null ? imp.getOriginalFileInfo().directory : "";
            File originalDir = new File(originalPath);
            File parentDir = originalDir.getParentFile();
            if (parentDir == null) {
                System.err.println("Could not determine the parent directory for: " + targetName);
                rm.reset();
                rm.close();
                continue;
            }

            String countedDirPath = parentDir.getAbsolutePath() + File.separator + originalDir.getName() + "_counted";
            new File(countedDirPath).mkdirs();

            String roisDirPath = countedDirPath + File.separator + "roi_images";
            new File(roisDirPath).mkdirs();

            String savePath = countedDirPath + File.separator + imp.getTitle();
            String roiImageSavePath = roisDirPath + File.separator + imp.getTitle();
            String roiSavePath = countedDirPath + File.separator + imp.getTitle() + "_rois.zip";

            new FileSaver(imp).saveAsTiff(savePath);
            new FileSaver(maskImage).saveAsTiff(roiImageSavePath);

            if (rm.getCount() > 0) {
                rm.runCommand("Save", roiSavePath);
            }

            rm.reset();
            rm.close();
        }

        return count;
    }

    private static String resolvePythonCommand() {
        for (String candidate : new String[]{"python3", "python"}) {
            try {
                Process process = new ProcessBuilder(candidate, "--version")
                        .redirectErrorStream(true)
                        .start();
                if (process.waitFor() == 0) {
                    return candidate;
                }
            } catch (IOException | InterruptedException ignored) {
            }
        }
        return null;
    }

    private static boolean isVirtualEnvExists(String envName) {
        File envDir = new File(envName);
        return envDir.exists() && envDir.isDirectory();
    }

    private static boolean createVirtualEnv(String pythonCommand, String envName) {
        try {
            Process process = new ProcessBuilder(pythonCommand, "-m", "venv", envName)
                    .redirectErrorStream(true)
                    .start();
            logProcessOutput(process);
            return process.waitFor() == 0;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean installPackages(String envName, String requirementsFile) {
        try {
            Process process = new ProcessBuilder(getPipPath(envName), "install", "-r", requirementsFile)
                    .redirectErrorStream(true)
                    .start();
            logProcessOutput(process);
            return process.waitFor() == 0;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean requirementsChanged() {
        try {
            String currentHash = hashFile(REQUIREMENTS);
            File hashFile = new File(REQUIREMENTS_HASH_FILE);
            if (!hashFile.exists()) {
                return true;
            }
            String storedHash = new String(Files.readAllBytes(hashFile.toPath())).trim();
            return !currentHash.equals(storedHash);
        } catch (Exception e) {
            return true;
        }
    }

    private static void updateRequirementsHash() {
        try {
            String currentHash = hashFile(REQUIREMENTS);
            try (PrintWriter out = new PrintWriter(new FileWriter(REQUIREMENTS_HASH_FILE))) {
                out.print(currentHash);
            }
        } catch (Exception e) {
            System.err.println("Failed to update requirements hash: " + e.getMessage());
        }
    }

    private static String hashFile(String path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] bytes = Files.readAllBytes(Paths.get(path));
        byte[] hashed = digest.digest(bytes);
        StringBuilder sb = new StringBuilder();
        for (byte b : hashed) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    private static String getPipPath(String envName) {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            return envName + "/Scripts/pip3.exe";
        }
        return envName + "/bin/pip";
    }

    private static String getPythonPath(String envName) {
        if (System.getProperty("os.name").toLowerCase().contains("win")) {
            return envName + "/Scripts/python.exe";
        }
        return envName + "/bin/python";
    }

    private static boolean runPythonScript(String envName, String scriptPath, String filepath, String batching) {
        return runPythonScriptAttempt(envName, scriptPath, filepath, batching, true);
    }

    private static boolean runPythonScriptAttempt(String envName, String scriptPath, String filepath,
                                                    String batching, boolean allowRetry) {
        try {
            Process process = new ProcessBuilder(getPythonPath(envName), scriptPath, filepath, batching)
                    .directory(new File(BASE_DIR))
                    .redirectErrorStream(true)
                    .start();
            logProcessOutput(process);
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return true;
            }
            if (allowRetry) {
                System.out.println("Python script failed. Reinstalling dependencies and retrying once...");
                installPackages(envName, REQUIREMENTS);
                updateRequirementsHash();
                return runPythonScriptAttempt(envName, scriptPath, filepath, batching, false);
            }
            IJ.error("Adipolabeler", "The Python script failed, including after a dependency repair attempt. "
                    + "Check the console log for details.");
            return false;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            IJ.error("Adipolabeler", "Failed to run the Python script: " + e.getMessage());
            return false;
        }
    }

    private static void logProcessOutput(Process process) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }
    }

    private void deleteDir(File file) {
        File[] contents = file.listFiles();
        if (contents != null) {
            for (File f : contents) {
                deleteDir(f);
            }
        }
        file.delete();
    }
}