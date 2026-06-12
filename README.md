# QuPath PhenoBIC extension

This extension adds support to run **PhenoBIC** (cell phenotype inference) within QuPath. It uses a Python environment to run a tile-based deep learning inference pipeline on multiplex images, while QuPath handles project management, cell segmentation, and measurement export/import.

![PhenoBIC Demo](media/PhenoBIC_GIF.gif)

---

## Citing

Please cite this extension by linking to this GitHub repository or to the release you used, and consider giving it a star ⭐️

If you use this extension in your work, please also cite the following:

- **QuPath**: Bankhead, P. et al. **QuPath: Open source software for digital pathology image analysis**. *Scientific Reports* (2017). <https://doi.org/10.1038/s41598-017-17204-5>

---

## System Requirements

- Windows Operating System
- Supports TIFF and TIFF-derived multiplex image formats (e.g. OME-TIFF)
- [QuPath v0.6.0](https://github.com/qupath/qupath/releases): You can try other versions but this is the version it has been fully tested on

---

## Installation

### Step 1: Install the PhenoBIC Python environment

You need a Python environment with the PhenoBIC dependencies. You can use conda:

1. Install [Anaconda](https://www.anaconda.com/).

2. Download [environment.yml](scripts/environment.yml)

3. In an Anaconda prompt terminal, get into the directory with the `environment.yml`
E.g.
```bash
cd C:\path_to_folder_containing_environment.yml 
```
Then create the Python enviornment with the required packages to run PhenoBIC
```bash
conda env create -f environment.yml  
```
Then activate the Python environment:
```bash
conda activate PhenoBIC 
```

It is not necessary to use a GPU for PhenoBIC inference but **for GPU acceleration**- you may need to install a GPU driver if you have not already. And then you need to install CUDA and cuDNN with conda to run tensorflow with GPU on Windows native.

E.g.
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 
```
Please refer to [tensorflow documentation](https://www.tensorflow.org/install/pip#windows-native_1) for additional instructions.

**Get the path to the Python executable**

The extension runs the Python script via this executable. Activate your environment and then get the path to the executable:

- (Anaconda Prompt):
  ```text
  conda activate PhenoBIC
  where python
  ```
  Example output: `C:/Users/.../envs/PhenoBIC_GPU/python.exe`


You will need this path for the script’s **REQUIRED CONFIG** (see below).

### Step 2: Install the QuPath PhenoBIC extension


- Download the latest `qupath-extension-phenobic-*v*-SNAPSHOT.jar` from [releases](https://github.com/Shachi-Mittal-Lab/PhenoBIC-qupath-extension/releases) and either place it in your QuPath extensions directory or drag & drop it onto the QuPath window and choose your user directory if prompted.

Restart QuPath after installing.

---

## Using the PhenoBIC extension

### Running cell phenotyping

1. **Open a project** and an image that has **detections** (i.e. cell segmentations).
2. Go to **Extensions → PhenoBIC → Run PhenoBIC → Cell phenotyping with PhenoBIC**.  
   The main script opens in the script editor.

3. Run the script:

 - **Edit the script config**  

    In the **REQUIRED CONFIG** block at the top, set:
   - `CHANNELS` – list of channel names (must match your image channel names in QuPath). Use **Extensions → PhenoBIC → Run PhenoBIC → Print Channel Names** to print the channel names for the current image.
   - `MODEL_PATH` – full path to `PhenoBIC_model1.keras` model. Download from [here](https://huggingface.co/mittal-research-lab/PhenoBIC/tree/main)
   - `PYTHON_SCRIPT` – full path to `PhenoBIC_backend.py`. Download the file from [here](scripts/PhenoBIC_backend.py).
   - `PYTHON_EXE` – path to the Python executable from Step 1.
   - `PREPROCESS_FIELD`: Set to `"whole image"` or `"TMA core"` depending on whether you would like normalization to be done on a full-image basis or separately for each TMA core (if there are TMA core Objects in the QuPath Project). We recommend core-level normalization when working with TMAs.
**WARNING: Using backslashes (`\`) in file paths will cause errors. Please only use forward slash (`/`)**

 - **Optional settings**  
    In the **OPTIONAL CONFIG** block, you can adjust other parameters to tailor your use of PhenoBIC.
   - `TILE_SIZE`: Size of square tiles in pixels used in the back-end, 10,000 pixels by default. For maximum speed, make this as large as possible without running into memory issues. Will affect speed of the run considerably.
   - `NUM_CELLS_BATCH`: Make this as large as possible without running into memory issues for maximum processing speed. 4,000 by default is fine. Prioritize changing `TILE_SIZE` to improve run speed.
   - `USE_GPU`: Set `true` to use GPU and `false` to not
   - `BUFFER_RATIO`: How much of the mutliplex signal around the cell to feed to the model. Recommended=0.1 → 10% buffering of cell bounding boxes.
   - `UPPER_CLIP_PERC`: Upper clip normalization parameter (recommended=0.9). This is the maximum of the <u>90<sup>th</sup></u> percentile within-cell channel intensity of all cells in the preprocessing field.
   - `LOWER_CLIP_PERC`: Lower clip normalization parameter (recommended=0.1). This is the minimum of the <u>10<sup>th</sup></u> percentile within-cell channel intensity of all cells in the preprocessing field.

 - **Run** – process the current image.

 - **Run for project** – batch process all images in the project.

 - **Save the scipt** - You can save the script with your specific configurations for easy access in another session.

### PhenoBIC outputs

Outputs are written under the project folder:

- `PhenoBIC_output/measurements/` – TSV files containing bounding box coordinates of each cell. Required for PhenoBIC predictions.
- `PhenoBIC_output/min_normalization/` – JSON files containing the lower clipping parameter for the raw intensity of each channel. Required for PhenoBIC predictions.
- `PhenoBIC_output/max_normalization/` – JSON files containing the upper clipping parameter for the raw intensity of each channel. Required for PhenoBIC predictions.
- `PhenoBIC_output/results/` – CSV files with per-channel phenotype classes. This can be used for further downstream single-cell quantitative and spatial analyses.
- QuPath detections are updated with measurements like `{ChannelName}_class` (1.0 = positive, 0.0 = negative). To visualize the cell expression class for any marker channel in the QuPath viewer, go to **Measure → Show measurement maps → {ChannelName}_class**.
![Visualization](imgs/visualize.png)

---

## Building from source

Build the extension with Gradle:

```bash
gradlew.bat clean build
```

Output JARs are in `build/libs/`.

---

## Notes and troubleshooting

- **No detections**  
  The script requires at least one detection object on the current image. Run cell detection (e.g. StarDist, Cellpose, native QuPath cell detection, etc.) before PhenoBIC.

- **Channel names**  
  `CHANNELS` must match the channel names in QuPath exactly. Use “Print Channel Names” to list them.

- **TMA cores**  
  If using `PREPROCESS_FIELD = "TMA core"`, the image must have a TMA grid and cores; detections are processed per core.

- **Python path**  
  If the subprocess fails, check that `PYTHON_EXE` is correct and that the same environment has the required packages (TensorFlow, tifffile, zarr, etc.). On some systems you may need to set or extend `PATH` so the executable is found.
**WARNING:** Using backslashes (`\`) in file paths will cause errors. Please only use forward slash (`/`)
  
- **GPU**  
  Set `USE_GPU = true` in the script only if your TensorFlow build and drivers support GPU; otherwise keep it `false` to use CPU.

- **Re-running PhenoBIC on an image**  
  If you need to re-run PhenoBIC, delete or rename the relevant files in `PhenoBIC_output` output directory within the project folder. Also go to **Measure → Show measurement manager** and delete the measurements **Bounds_x**, **Bounds_y**, **Bounds_width**, **Bounds_height**, **{ChannelName}_LowPercentile**, **{ChannelName}_HighPercentile**, and **{ChannelName}_class** so that QuPath knows to re-run PhenoBIC for those channels.

---
