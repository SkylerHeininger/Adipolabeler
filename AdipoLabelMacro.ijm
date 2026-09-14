inputDir = getDirectory("Choose a folder");

Dialog.create("Adipolabeler Batch Settings");
Dialog.addCheckbox("Override script defaults (config.py) with settings below", false);
Dialog.addCheckbox("Enable bias-correction scaling", false);
Dialog.addNumber("Scaling factor", 0.75);
Dialog.addNumber("Sphericity threshold (0-1)", 0.5);
Dialog.addNumber("Cell size threshold (pixels^2)", 750);
Dialog.addNumber("Batch size (0 = auto-detect)", 0);
Dialog.addCheckbox("Save these settings to file for future runs", false);
Dialog.addCheckbox("Reset to script defaults (clear any previously saved settings)", false);
Dialog.show();

overrideDefaults = Dialog.getCheckbox();

useScalingRaw = Dialog.getCheckbox();
if (useScalingRaw == 1) {
    useScaling = "true";
} else {
    useScaling = "false";
}

scaleFactor = Dialog.getNumber();
sphericity = Dialog.getNumber();
cellSize = Dialog.getNumber();
batchSize = d2s(Dialog.getNumber(), 0);

saveRaw = Dialog.getCheckbox();
if (saveRaw == 1) {
    save = "true";
} else {
    save = "false";
}

resetRaw = Dialog.getCheckbox();
if (resetRaw == 1) {
    reset = "true";
} else {
    reset = "false";
}

runOptions = "arg=" + inputDir + "\nbatch=true\nsave=" + save + "\nreset=" + reset;
if (overrideDefaults == 1) {
    runOptions = runOptions + "\nscaling=" + useScaling + "\nscale_factor=" + scaleFactor
        + "\nsphericity=" + sphericity + "\ncell_size=" + cellSize + "\nbatch_size=" + batchSize;
}

run("Adipo Label", runOptions);
