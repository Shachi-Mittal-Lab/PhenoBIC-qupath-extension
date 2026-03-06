/**
 * PhenoBIC cell phenotype inference — run from Script Editor for live log.
 * Uses project folder PhenoBIC_output for results export.
 * Supports multiple CHANNELS; output CSV has one column per channel (channel name as header).
 * Use "Run" for current image, or "Run for project" to batch process all images.
 *
 * Edit the REQUIRED CONFIG block below, then Run (or Run for project).
 **/

// ═══════════════════════════════════════════════════════════════════════════
// REQUIRED CONFIG — edit these for your setup
// ═══════════════════════════════════════════════════════════════════════════
def CHANNELS = ["CD3 (Opal 480)", "CD8 (Opal 780)"] // list of channel to run PhenoBIC (must match image channel names in QuPath)
def MODEL_PATH = "D:/8MyLym/phen_classifiers/manu_classifiers/PhenoBIC_model1.keras" // local path to PhenoBIC model
def PYTHON_SCRIPT = "C:/Users/asankar6/Desktop/PhenoBIC-qupath-extension/scripts/PhenoBIC_backend.py" // local path to PhenoBIC python script
def PYTHON_EXE = "C:/Users/asankar6/AppData/Local/anaconda3/envs/PhenoBIC/python.exe" // local path to PhenoBIC python executable
def PREPROCESS_FIELD = "whole image" // "whole image" = whole-image percentiles; "TMA core" = percentiles computed per TMA core
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// OPTIONAL CONFIG
// ═══════════════════════════════════════════════════════════════════════════
def TILE_SIZE = 10000 // Square tile size in pixels. For maximum speed --> set as large as possible without running into memory issues.
def NUM_CELLS_BATCH = 4000 // For maximum multiprocessing speed --> set as large as possible without running into memory issues
def USE_GPU = true // Set false to not use GPU
def BUFFER_RATIO = 0.1 // Recommended to use ten percent (0.1) buffered cell bounding box of cell expression classification
def UPPER_CLIP_PERC = 0.9 // Recommended upper clip normalization parameter = 0.9 (highest within-cell 90th percentile intensity of all cells)
def LOWER_CLIP_PERC = 0.1// Recommended lower clip normalization parameter = 0.1 (lowest within-cell 10th percentile intensity of all cells)
// ═══════════════════════════════════════════════════════════════════════════



// ═══════════════════════════════════════════════════════════════════════════
// Pipeline - do not edit
// ═══════════════════════════════════════════════════════════════════════════
//(1) Imports and data processing set-up
import qupath.lib.regions.RegionRequest
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObjectTools
import java.net.URI
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Comparator
import java.util.Collections
import java.util.Arrays

// For easy reference throughout the script
def BOUNDS_X = "Bounds_x"
def BOUNDS_Y = "Bounds_y"
def BOUNDS_WIDTH = "Bounds_width"
def BOUNDS_HEIGHT = "Bounds_height"
def OBJECT_ID = "Object ID"
def PHENOBIC_OUTPUT = "PhenoBIC_output"
def MEASUREMENTS_SUBDIR = "measurements"
def RESULTS_SUBDIR = "results"
def MIN_NORMALIZATION_SUBDIR = "min_normalization"
def MAX_NORMALIZATION_SUBDIR = "max_normalization"
def LOW_PERCENTILE_SUFFIX = "_LowPercentile"
def HIGH_PERCENTILE_SUFFIX = "_HighPercentile"

// Consistent method of logging
void log(String msg) {
    println "[PhenoBIC] ${msg}"
}

// Get the current projet directory path
def project = getProject()
if (project == null) {
    log "No project open. Open a project first."
    return
}
def projectPath = project.getPath()
if (projectPath == null || (projectPath instanceof String && projectPath.isEmpty())) {
    log "Project has no path (unsaved?). Save the project first."
    return
}
// Get the Path object if it is a string or other datatype
def projectDir = (projectPath instanceof java.nio.file.Path)
    ? projectPath.getParent()
    : Paths.get(projectPath.toString()).getParent()
// Output root directory
def phenoBICRoot = projectDir.resolve(PHENOBIC_OUTPUT)
// Output directory for generated measurements that are needed for PhenoBIC inference
def measurementsDir = phenoBICRoot.resolve(MEASUREMENTS_SUBDIR)
// Output directory for PhenoBIC results
def resultsDir = phenoBICRoot.resolve(RESULTS_SUBDIR)
// Directories for normalization JSON (min/max per channel)
def minNormalizationDir = phenoBICRoot.resolve(MIN_NORMALIZATION_SUBDIR)
def maxNormalizationDir = phenoBICRoot.resolve(MAX_NORMALIZATION_SUBDIR)
// Creating the output root and subdirectories
[phenoBICRoot, measurementsDir, resultsDir, minNormalizationDir, maxNormalizationDir].each { Files.createDirectories(it) }
log "Output root: ${phenoBICRoot}"

// Get the imageData of the open image
def imageData = getCurrentImageData()
if (imageData == null) {
    log "No image open. Open an image or use Run for project."
    return
}

// Get image metadata
def server = imageData.getServer()
def meta = server.getMetadata()
def name = meta.getName()
// Get segmented detections/cells
def detections = imageData.getHierarchy().getDetectionObjects()
if (detections.isEmpty()) {
    log "No detections for ${name}. Add detections and run again."
    return
}
// Get channel names from metadata
def channelNames = meta.getChannels()?.collect { it.getName() } ?: []
if (CHANNELS == null || !(CHANNELS instanceof Collection) || CHANNELS.isEmpty()) {
    log "CHANNELS must be a non-empty list of channel names. Available: ${channelNames}"
    return
}

// (2) Generate ROI bounds if not already present
// Get the list of existing detection measurements
def firstDet = detections[0]
def ml0 = firstDet.getMeasurementList()
// Checking if the first detection has the needed measurements about bounding box
boolean hasBounds = ml0.containsKey(BOUNDS_X) && ml0.containsKey(BOUNDS_Y) && ml0.containsKey(BOUNDS_WIDTH) && ml0.containsKey(BOUNDS_HEIGHT)
// Generating the bounding box measurements if not already present and updating object hierarchy
if (!hasBounds) {
    log "Adding ROI bounds measurements (Bounds_x, Bounds_y, Bounds_width, Bounds_height)..."
    for (cell in detections) {
        def roi = cell.getROI()
        cell.getMeasurementList().put(BOUNDS_X, roi.getBoundsX())
        cell.getMeasurementList().put(BOUNDS_Y, roi.getBoundsY())
        cell.getMeasurementList().put(BOUNDS_WIDTH, roi.getBoundsWidth())
        cell.getMeasurementList().put(BOUNDS_HEIGHT, roi.getBoundsHeight())
    }
    imageData.getHierarchy().fireHierarchyChangedEvent(imageData.getHierarchy(), imageData.getHierarchy().getRootObject())
    log "Bounds measurements added."
}

// Resolve image path and output file paths (used by both whole image and TMA-core paths)
def imagePath = null
// Get image path on local disk
def uris = server.getURIs()
if (uris && !uris.isEmpty()) {
    def first = uris.iterator().next()
    if (first?.scheme == "file") imagePath = Paths.get(first)
}
if (imagePath == null && server.getPath()) {
    try { imagePath = Paths.get(URI.create(server.getPath())) } catch (e) {}
    if (imagePath == null) imagePath = Paths.get(server.getPath().trim()).toAbsolutePath().normalize()
}
if (imagePath == null) {
    def entry = project.getImageList().find { it.getImageName() == name }
    if (entry?.getEntryPath() != null) imagePath = entry.getEntryPath()
}
if (imagePath == null || !Files.isRegularFile(imagePath)) {
    log "Image file path not found for ${name}."
    return
}
// Output file names
def imageName = name.contains(".") ? name.substring(0, name.lastIndexOf(".")) : name
def measurementsFile = measurementsDir.resolve("${imageName}.tsv")
def outputCsv = resultsDir.resolve("${imageName}.csv")

// Path helpers and Python setup (used by both paths)
def pathForSubprocess = { String s ->
    if (!s?.trim()) return s
    try { return Paths.get(s.trim()).toAbsolutePath().normalize().toString().replace('\\', '/') }
    catch (e) { return s.trim() }
}
// Safe channel name to be used in the code-block below, especially for writing in filename
def safeFilename(String ch) {
    ch?.replaceAll(/[^\w\-.]/, "_") ?: "channel"
}
def safeCoreName(core) {
    core?.getName()?.replaceAll(/[^\w\-.]/, "_") ?: "core"
}
// Python-safe subprocess path arguments
def scriptNorm = pathForSubprocess(PYTHON_SCRIPT)
def imageNorm = pathForSubprocess(imagePath.toString())
def modelNorm = pathForSubprocess(MODEL_PATH)
def pythonExe = (PYTHON_EXE?.trim() ?: "python").trim()

// Check if the PREPROCESS_FIELD is valid
if (PREPROCESS_FIELD != "whole image" && PREPROCESS_FIELD != "TMA core") {
    log "PREPROCESS_FIELD must be 'whole image' or 'TMA core'. Got: ${PREPROCESS_FIELD}"
    return
}

def channelResults = []

// Whole-image path: percentiles computed in Python from cell bounding boxes
if (PREPROCESS_FIELD == "whole image") {
// (3) Build channel list (name + index in multiplex array)
def channelInfo = []
for (def chName : CHANNELS) {
    def channelIndex = channelNames.indexOf(chName)
    // Handling incorrect channel name
    if (channelIndex < 0) {
        log "Channel '${chName}' not found. Available: ${channelNames}"
        return
    }
    channelInfo << [name: chName, index: channelIndex]
}

// (4) Write measurements TSV file for cell bounds
log "Writing measurements to ${measurementsFile}..."
def headerCols = ['Image', 'Object ID', BOUNDS_X, BOUNDS_Y, BOUNDS_WIDTH, BOUNDS_HEIGHT] as List
// Writing the measurements TSV file. Will overwrite existing file
measurementsFile.toFile().withPrintWriter { w ->
    w.println(headerCols.join("\t"))
    for (def obj : detections) {
        // Get the all measurements in the row of the dataframe corresponding the Cell ID
        def ml = obj.getMeasurementList()
        def oid = ml.containsKey(OBJECT_ID) ? String.valueOf(ml.get(OBJECT_ID).longValue()) : String.valueOf(obj.getID())
        def x = ml.containsKey(BOUNDS_X) ? ml.get(BOUNDS_X) : 0
        def y = ml.containsKey(BOUNDS_Y) ? ml.get(BOUNDS_Y) : 0
        def boundsW = ml.containsKey(BOUNDS_WIDTH) ? ml.get(BOUNDS_WIDTH) : 0
        def h = ml.containsKey(BOUNDS_HEIGHT) ? ml.get(BOUNDS_HEIGHT) : 0
        // Write the row as a tab-delimited string
        w.println([name, oid, x, y, boundsW, h].join("\t"))
    }
}
log "Measurements written."

def measurementsNorm = pathForSubprocess(measurementsFile.toString())

// (5) Run Python for each channel; collect per-channel CSVs then merge
// iterate over channels
for (def info : channelInfo) {
    // Get the name, output file path
    def chName = info.name
    def channelCsv = resultsDir.resolve("${imageName}_${safeFilename(chName)}.csv")
    def outputNorm = pathForSubprocess(channelCsv.toString())
    // CLI arguments
    def minJsonPath = minNormalizationDir.resolve("${imageName}.json")
    def maxJsonPath = maxNormalizationDir.resolve("${imageName}.json")
    def cmd = [pythonExe, scriptNorm,
        "--image", imageNorm,
        "--measurements-tsv", measurementsNorm,
        "--channel-index", String.valueOf(info.index),
        "--channel-name", chName,
        "--output-min-json", pathForSubprocess(minJsonPath.toString()),
        "--output-max-json", pathForSubprocess(maxJsonPath.toString()),
        "--lower-percentile", String.valueOf(100 * LOWER_CLIP_PERC),
        "--upper-percentile", String.valueOf(100 * UPPER_CLIP_PERC),
        "--buffer-ratio", String.valueOf(BUFFER_RATIO),
        "--output-csv", outputNorm,
        "--model-path", modelNorm,
        "--num-cells-batch", String.valueOf(NUM_CELLS_BATCH),
        "--tile-size", String.valueOf(TILE_SIZE)
    ]
    if (!USE_GPU) cmd += "--no-gpu"
    // Build process launcher with the arguments and show outputs in log
    def pb = new ProcessBuilder(cmd).redirectErrorStream(true)
    if (PYTHON_EXE?.trim()) {
        // Normalize path
        def exePath = Paths.get(PYTHON_EXE.trim().replace('/', File.separator))
        // verify that the path exists and is a "regular file"
        if (Files.isRegularFile(exePath)) {
            def envRoot = exePath.getParent()
            // Use correct separator according to OS
            def pathSep = System.getenv("PATH")?.contains(";") ? ";" : ":"
            // Add to PATH
            def extra = [envRoot.toString(), envRoot.resolve("Library").resolve("bin").toString(), envRoot.resolve("Scripts").toString()]
            def existingPath = pb.environment().get("PATH") ?: ""
            pb.environment().put("PATH", extra.join(pathSep) + pathSep + existingPath)
        }
    }
    log "Starting Python: channel '${chName}' (index ${info.index})"
    def proc = pb.start()
    // Print output in consol/log
    proc.inputStream.newReader().eachLine { line -> println line }
    int exitCode = proc.waitFor()
    // Error in code run
    if (exitCode != 0) {
        log "Python exited with code ${exitCode} for channel '${chName}'"
        return
    }
    if (!Files.exists(channelCsv)) {
        log "Output CSV not created: ${channelCsv}"
        return
    }
    // Create empty map
    def oidToClass = [:]
    // Read all lines of CSV file with PhenoBIC outputs
    def lines = channelCsv.toFile().readLines("UTF-8")
    // Zeroth line is the column header
    for (int i = 1; i < lines.size(); i++) {
        // Comma delimited values
        def parts = lines[i].split(",", -1)
        // Checking at least two columns exist
        if (parts.length >= 2) {
            // Remove white space and string quotes
            def oid = parts[0].trim().replaceAll('^"|"$', "")
            def cls = parts[1].trim().replaceAll('^"|"$', "")
            // Assing class to object ID
            oidToClass[oid] = cls
        }
    }
    // Channels list with associated cell ID --> classification mapping recorded
    channelResults << [channelName: chName, oidToClass: oidToClass]
    // Delete temp file
    try { Files.deleteIfExists(channelCsv) } catch (e) {}   // optional: remove temp per-channel CSV
}

// Normalize percentiles for the TMA cores
} else if (PREPROCESS_FIELD == "TMA core") {
// (3)–(5) per TMA core: percentiles per core, write TSV per core, run Python per channel per core, merge results of all channels into one CSV file
    def hierarchy = imageData.getHierarchy()
    def tmaGrid = hierarchy.getTMAGrid()
    // Check if the TMA grid is found
    if (tmaGrid == null) {
        log "No TMA grid found. Use PREPROCESS_FIELD = 'whole image' or add a TMA grid to the image."
        return
    }
    // Get list of TMA cores
    def coreList = tmaGrid.getTMACoreList()
    if (coreList == null || coreList.isEmpty()) {
        log "TMA grid has no cores."
        return
    }
    // Initialize channelResults: one map per channel to merge per-core results
    for (def chName : CHANNELS) {
        channelResults << [channelName: chName, oidToClass: [:]]
    }
    for (def core : coreList) {
        // Use spatial containment: detections whose centroid is inside this core's ROI
        // since hierarchy may not link cores to cells
        def coreRoi = core.getROI()
        if (coreRoi == null) {
            log "Core ${core.getName()} has no ROI; skipping."
            continue
        }
        def coreDetections = PathObjectTools.filterByROIContainsCentroid(coreRoi, detections)
        if (coreDetections == null || coreDetections.isEmpty()) {
            log "Core ${core.getName()} has no detections; skipping."
            continue
        }
        def coreName = safeCoreName(core)
        log "Processing TMA core ${core.getName()} (${coreDetections.size()} cells)"
        // (3) Build channel list (name + index in multiplex array)
        def channelInfoCore = []
        for (def chName : CHANNELS) {
            def channelIndex = channelNames.indexOf(chName)
            if (channelIndex < 0) {
                log "Channel '${chName}' not found. Available: ${channelNames}"
                return
            }
            channelInfoCore << [name: chName, index: channelIndex]
        }
        // (4) Write core-specific TSV
        def coreTsvPath = measurementsDir.resolve("${imageName}_core_${coreName}.tsv")
        def headerColsCore = ['Image', 'Object ID', BOUNDS_X, BOUNDS_Y, BOUNDS_WIDTH, BOUNDS_HEIGHT] as List
        coreTsvPath.toFile().withPrintWriter { w ->
            w.println(headerColsCore.join("\t"))
            // Writing the measurements TSV file. Will overwrite existing file
            for (def obj : coreDetections) {
                // Get the all measurements in the row of the dataframe corresponding the Cell ID
                def ml = obj.getMeasurementList()
                def oid = ml.containsKey(OBJECT_ID) ? String.valueOf(ml.get(OBJECT_ID).longValue()) : String.valueOf(obj.getID())
                def x = ml.containsKey(BOUNDS_X) ? ml.get(BOUNDS_X) : 0
                def y = ml.containsKey(BOUNDS_Y) ? ml.get(BOUNDS_Y) : 0
                def boundsW = ml.containsKey(BOUNDS_WIDTH) ? ml.get(BOUNDS_WIDTH) : 0
                def h = ml.containsKey(BOUNDS_HEIGHT) ? ml.get(BOUNDS_HEIGHT) : 0
                // Write the row as a tab-delimited string
                w.println([name, oid, x, y, boundsW, h].join("\t"))
            }
        }
        def coreMeasurementsNorm = pathForSubprocess(coreTsvPath.toString())
        // (5) Run Python for each channel for this core; merge into channelResults
        for (def info : channelInfoCore) {
            def chName = info.name
            def channelIdx = CHANNELS.indexOf(chName)
            def channelCsv = resultsDir.resolve("${imageName}_core_${coreName}_${safeFilename(chName)}.csv")
            def outputNorm = pathForSubprocess(channelCsv.toString())
            def minJsonCore = minNormalizationDir.resolve("${imageName}_core_${coreName}.json")
            def maxJsonCore = maxNormalizationDir.resolve("${imageName}_core_${coreName}.json")
            // CLI arguments
            def cmd = [pythonExe, scriptNorm,
                "--image", imageNorm,
                "--measurements-tsv", coreMeasurementsNorm,
                "--channel-index", String.valueOf(info.index),
                "--channel-name", chName,
                "--output-min-json", pathForSubprocess(minJsonCore.toString()),
                "--output-max-json", pathForSubprocess(maxJsonCore.toString()),
                "--lower-percentile", String.valueOf(100 * LOWER_CLIP_PERC),
                "--upper-percentile", String.valueOf(100 * UPPER_CLIP_PERC),
                "--buffer-ratio", String.valueOf(BUFFER_RATIO),
                "--output-csv", outputNorm,
                "--model-path", modelNorm,
                "--num-cells-batch", String.valueOf(NUM_CELLS_BATCH),
                "--tile-size", String.valueOf(TILE_SIZE)
            ]
            // Add the --no-gpu flag if the USE_GPU flag is false
            if (!USE_GPU) cmd += "--no-gpu"
            // Create a new process builder with the command
            def pb = new ProcessBuilder(cmd).redirectErrorStream(true)
            // If the PYTHON_EXE flag is set, add the PYTHON_EXE to the PATH environment variable
            if (PYTHON_EXE?.trim()) {
                def exePath = Paths.get(PYTHON_EXE.trim().replace('/', File.separator))
                if (Files.isRegularFile(exePath)) {
                    def envRoot = exePath.getParent()
                    def pathSep = System.getenv("PATH")?.contains(";") ? ";" : ":"
                    def extra = [envRoot.toString(), envRoot.resolve("Library").resolve("bin").toString(), envRoot.resolve("Scripts").toString()]
                    def existingPath = pb.environment().get("PATH") ?: ""
                    pb.environment().put("PATH", extra.join(pathSep) + pathSep + existingPath)
                }
            }
            log "Starting Python: core ${coreName}, channel '${chName}'"
            // Start the process
            def proc = pb.start()
            // Print the output in the console/log
            proc.inputStream.newReader().eachLine { line -> println line }
            int exitCode = proc.waitFor()
            // Error in code run
            if (exitCode != 0) {
                log "Python exited with code ${exitCode} for core ${coreName}, channel '${chName}'"
                return
            }
            if (!Files.exists(channelCsv)) {
                log "Output CSV not created: ${channelCsv}"
                return
            }
            // Create an empty map for the object ID to class mapping
            def oidToClass = [:]
            // Read all lines of the CSV file with the PhenoBIC outputs
            def lines = channelCsv.toFile().readLines("UTF-8")
            // Iterate over all the lines in the CSV file
            for (int i = 1; i < lines.size(); i++) {
                def parts = lines[i].split(",", -1)
                if (parts.length >= 2) {
                    def oid = parts[0].trim().replaceAll('^"|"$', "")
                    def cls = parts[1].trim().replaceAll('^"|"$', "")
                    // Assign the class to the object ID
                    oidToClass[oid] = cls
                }
            }
            // Update the channel results with the object ID to class mapping
            channelResults[channelIdx].oidToClass.putAll(oidToClass)
            // Delete the temporary CSV file
            try { Files.deleteIfExists(channelCsv) } catch (e) {}
        }
    }
}

// (6) Write combined CSV: Object ID, CHANNEL1, CHANNEL2, ... (channel name as column header)
// Union set of all object IDs
def allOids = new LinkedHashSet()
for (def r : channelResults) {
    allOids.addAll(r.oidToClass.keySet())
}
// Write classification output CSV file
outputCsv.toFile().withWriter("UTF-8") { w ->
    // Write header line
    w.println("Object ID," + (CHANNELS.collect { "\"${it}\"" }.join(",")))
    // Write each row
    for (def oid : allOids) {
        // Start with object ID
        def row = [oid]
        // iterate over channels
        for (def r : channelResults) {
            // Get the class of each row in the channel
            def cls = r.oidToClass[oid]
            // Append it to the row for the channel
            row << (cls != null ? cls : "")
        }
        // Write the row and then move on to the next one
        w.println(row.collect { "\"${it}\"" }.join(","))
    }
}
log "Combined results written to ${outputCsv}"

// (7) Add cell expression classification measurements in QuPath: for each channel, "{channelName}_class" = 1.0 or 0.0
// Iterate over each cell
for (def obj : detections) {
    // Get the list of measurements and object ID
    def ml = obj.getMeasurementList()
    def oid = ml.containsKey(OBJECT_ID) ? String.valueOf(ml.get(OBJECT_ID).longValue()) : String.valueOf(obj.getID())
    // Iterate over each channel
    for (def r : channelResults) {
        // Get the cell expression classificaiton result
        def cls = r.oidToClass[oid]
        if (cls != null) {
            // One-hot encode the cell expression classification result
            double val = "pos".equalsIgnoreCase(cls) ? 1.0 : 0.0
            // Update the measurements with the classification
            ml.put("${r.channelName}_class", val)
        }
    }
    ml.close()
}
// Update QuPath's object hierarchy with new cell expression classification measurements
imageData.getHierarchy().fireHierarchyChangedEvent(imageData.getHierarchy(), imageData.getHierarchy().getRootObject())
log "Done. Measurements updated: ${CHANNELS.collect { "${it}_class" }}"
