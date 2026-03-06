"""
PhenoBIC cell phenotype inference from multiplex images.
Back-end script for the QuPath extension.

Reads TIFF  via tifffile + zarr.
Tile-by-tile processing
Reads pre-exported measurements TSV (bounds + normalization parameters). Uses a
multiprocessing pool to create an array of cell bounding boxcrops per batch followed
by PhenoBIC inference for each cell crop.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO and WARNING messages
import sys

# Disable GPU before TensorFlow is imported (e.g. --no-gpu from QuPath/Groovy).
if "--no-gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import tifffile
import zarr



# OME-TIFF axis and series selection

def _ometiff_zarr_axes(shape):
    """Infer (y_axis, x_axis, c_axis) from array shape."""
    shape = tuple(shape)
    nd = len(shape)
    if nd == 2:
        return 0, 1, None
    c_axis = int(np.argmin(shape))
    big_two = np.argsort(shape)[-2:]
    y_axis = int(min(big_two))
    x_axis = int(max(big_two))
    return y_axis, x_axis, c_axis


def _axes_from_series_axes(axes, shape):
    """Get (y_axis, x_axis, c_axis) from tifffile series.axes; fallback to shape heuristic."""
    # Read series from tifffile to find the index in the dimensionality of X, Y, and channel
    if axes and "Y" in axes and "X" in axes:
        # Get the index of the Y, X, and channel axes (C or S) and return
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        if "C" in axes:
            c_axis = axes.index("C")
        elif "S" in axes:
            c_axis = axes.index("S")
        else:
            c_axis = None
        return y_axis, x_axis, c_axis
    return _ometiff_zarr_axes(shape)


def _series_index_with_largest_yx(tif):
    """Return the series index whose shape has the largest Y*X (spatial extent)."""
    # The idea of this function is to find the highest resolution series
    if not tif.series:
        raise ValueError("No series in TIFF")
    best_i = 0
    best_size = 0
    # Iteratre over all the series
    for i, s in enumerate(tif.series):
        shape = s.shape
        axes = getattr(s, "axes", None)
        y_axis, x_axis, _ = _axes_from_series_axes(axes, shape)
        if len(shape) > max(y_axis, x_axis):
            size = int(shape[y_axis]) * int(shape[x_axis])
            if size > best_size:
                best_size = size
                best_i = i
    return best_i


def _ometiff_shape_and_axes(image_path):
    """Read OME-TIFF shape and (y_axis, x_axis, c_axis) without loading pixels. Uses series with largest Y*X."""
    with tifffile.TiffFile(image_path) as tif:
        if not tif.series:
            raise ValueError(f"No series in TIFF: {image_path}")
        # Find the highest resolution series
        idx = _series_index_with_largest_yx(tif)
        # Lazy load the series
        series = tif.series[idx]
        shape = series.shape
        axes = getattr(series, "axes", None)
    # Get the axes of the series that correspond to Y, X, and channel
    y_axis, x_axis, c_axis = _axes_from_series_axes(axes, shape)
    return shape, y_axis, x_axis, c_axis


# Tile-based channel reading (tifffile + zarr)

def _slice_for_axis(i, y_axis, x_axis, c_axis, channel_index, y_lo, y_hi, x_lo, x_hi):
    """Return the slice for axis i: Y, X, or channel dimension."""
    if i == y_axis:
        return slice(y_lo, y_hi) if (y_lo is not None and y_hi is not None) else slice(None)
    if i == x_axis:
        return slice(x_lo, x_hi) if (x_lo is not None and x_hi is not None) else slice(None)
    if c_axis is not None and i == c_axis:
        return channel_index
    return 0


def _read_channel_from_ometiff_zarr(
    image_path, channel_index, y_axis, x_axis, c_axis,
    y_lo=None, y_hi=None, x_lo=None, x_hi=None,
):
    """
    Read one channel from OME-TIFF via tifffile+zarr.
    Optional window [y_lo:y_hi, x_lo:x_hi]. Uses series and (if Group) array with largest Y*X.
    """
    # Read (lazily) the high-res image from the image file
    with tifffile.TiffFile(image_path) as tif:
        idx = _series_index_with_largest_yx(tif)
        store = tif.series[idx].aszarr()
        root = zarr.open(store, mode="r")

        # If its an array this is the final channel.
        if hasattr(root, "ndim"):
            arr = root
        # If its a Group object, we need to find the highest resolution array
        else:
            # Get the list of arrays in the group
            keys = list(getattr(root, "array_keys", lambda: list(root.keys()))())
            if not keys:
                raise ValueError("No array in OME-TIFF zarr store")
            # Iterate over all the arrays in the group and find the largest X*Y (highest resolution)
            best_key = keys[0]
            best_size = 0
            for k in keys:
                a = root[k]
                if hasattr(a, "shape") and a.ndim > max(y_axis, x_axis):
                    size = int(a.shape[y_axis]) * int(a.shape[x_axis])
                    if size > best_size:
                        best_size = size
                        best_key = k
            arr = root[best_key]

        # Dimensions of the high res image array
        nd = arr.ndim
        # Create a tuple of slices for the Y, X, and channel dimensions
        idx = tuple(
            _slice_for_axis(i, y_axis, x_axis, c_axis, channel_index, y_lo, y_hi, x_lo, x_hi)
            for i in range(nd)
        )
        # Read the sliced array into a numpy array
        out = np.asarray(arr[idx], dtype=np.float64)
    # Squeeze the array to remove any singleton dimensions
    return np.squeeze(out)



# Per-worker state and ROI extraction

_worker_channel_plane = None
_worker_y_axis = None
_worker_x_axis = None


def _plane_yx_axes(y_axis, x_axis):
    """Map full-array Y/X axis indices to 2D plane indices (0, 1)."""
    if y_axis < x_axis:
        return 0, 1  # plane is (Y, X)
    return 1, 0  # plane is (X, Y)


def _init_worker_tile(image_path, channel_index, y_lo, y_hi, x_lo, x_hi, y_axis, x_axis, c_axis):
    """Load the tile [y_lo:y_hi, x_lo:x_hi] once per worker. Sets global plane and axis indices."""
    global _worker_channel_plane, _worker_y_axis, _worker_x_axis
    _worker_channel_plane = _read_channel_from_ometiff_zarr(
        image_path, channel_index, y_axis, x_axis, c_axis,
        y_lo=y_lo, y_hi=y_hi, x_lo=x_lo, x_hi=x_hi,
    )
    _worker_y_axis, _worker_x_axis = _plane_yx_axes(y_axis, x_axis)


def _extract_roi(bounds):
    """Crop the worker's channel plane for one cell.
    Bounds are (x, y, w, h, x_buf, y_buf) in tile-relative coordinates.
    x and y are upper-left coordinates of the cell bounding box."""
    global _worker_channel_plane, _worker_y_axis, _worker_x_axis
    plane = _worker_channel_plane
    ya, xa = _worker_y_axis, _worker_x_axis
    x, y, w, h, x_buf, y_buf = bounds
    # Calculate the lower and upper bounds of the cell bounding box in the channel plane
    # Including buffering by a factor of buffer ratio (as show by x_buf and y_buf)
    y_lo = max(0, int(y - y_buf))
    y_hi = min(plane.shape[ya], int(y + h + y_buf))
    x_lo = max(0, int(x - x_buf))
    x_hi = min(plane.shape[xa], int(x + w + x_buf))
    s = [slice(None), slice(None)]
    # Generate and X and Y slices for the crop of the cell
    s[ya] = slice(y_lo, y_hi)
    s[xa] = slice(x_lo, x_hi)
    # Return the cropped channel plane
    return np.asarray(plane[tuple(s)])


def preprocess_roi(roi, min_val, max_val):
    """Linearly scale ROI to [0,255] and replicate to 3 channels for the model."""
    scaled = np.clip((roi.astype(np.float64) - min_val) / (max_val - min_val), 0, 1)
    return np.stack((np.uint8(scaled * 255),) * 3, axis=-1)


# Tile assignment and main pipeline

def _log(msg):
    """Print progress messages to the console."""
    print(msg, flush=True)


def _write_normalization_json(json_path, channel_name, value):
    """Write the normalization dictionary JSON file"""
    data = {}
    # If the JSON file exists, load the data from the file
    # Allows appending norm params for new channel to existing JSON file
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    # Key=channel name and value=normalization value
    data[channel_name] = value
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    # Write JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _tile_contains_box(x_lo, y_lo, x_hi, y_hi, bx_lo, by_lo, bx_hi, by_hi):
    """Does the fully buffered cell bounding box fit within the tile?"""
    return x_lo <= bx_lo and bx_hi <= x_hi and y_lo <= by_lo and by_hi <= y_hi


def _assign_cell_to_tile(cx, cy, bx_lo, by_lo, bx_hi, by_hi, n_ty, n_tx, step_y, step_x, tile_size, height, width):
    """Return index of the tile that contains the cell centroid and its full buffered bounding box."""
    # Iterate over all the tiles in the grid
    for ty in range(n_ty):
        for tx in range(n_tx):
            # Calculate the lower and upper bounds of the tile
            y_lo = ty * step_y
            y_hi = min(ty * step_y + tile_size, height)
            x_lo = tx * step_x
            x_hi = min(tx * step_x + tile_size, width)
            # Check if the cell centroid and its full buffered bounding box fit within the tile
            if x_lo <= cx < x_hi and y_lo <= cy < y_hi and _tile_contains_box(x_lo, y_lo, x_hi, y_hi, bx_lo, by_lo, bx_hi, by_hi):
                # Return the index of the tile if it does
                return (ty, tx)
    # If no tile is found, return (0, 0)
    return (0, 0)


def _process_tile_for_percentiles(args):
    """Worker: read one tile, compute per-cell percentiles for all cells in tile.
    Returns [(cell_idx, low, high), ...]."""
    # Get the arguments
    (image_path, channel_index, y_axis, x_axis, c_axis, y_lo, y_hi, x_lo, x_hi,
     cell_list, lower_percentile, upper_percentile) = args
    # If no cells in the tile, return an empty list
    if not cell_list:
        return []
    # Read the tile from the image channel
    tile = _read_channel_from_ometiff_zarr(
        image_path, channel_index, y_axis, x_axis, c_axis,
        y_lo=y_lo, y_hi=y_hi, x_lo=x_lo, x_hi=x_hi,
    )
    # Get the Y and X axes
    ya, xa = _plane_yx_axes(y_axis, x_axis)
    # Initialize empty output list
    results = []
    # Iterate over all the cells in the tile
    for (i, x, y, w, h) in cell_list:
        # Calculate the lower and upper bounds of the cell in the tile relative coordinates
        ry_lo, ry_hi = y - y_lo, y - y_lo + h
        rx_lo, rx_hi = x - x_lo, x - x_lo + w
        # Create a slice for the cell in the tile
        s = [slice(None), slice(None)]
        s[ya] = slice(ry_lo, ry_hi)
        s[xa] = slice(rx_lo, rx_hi)
        # Read the cell bbox region
        roi = np.asarray(tile[tuple(s)])
        # Flatten the array of channel signal in the cell bbox
        flat = np.asarray(roi).ravel()
        # Skip if no pixels in the cell bbox
        if flat.size == 0:
            continue
        # Compute the lower and upper percentiles of the channel signal in the cell bbox
        low = float(np.percentile(flat, lower_percentile))
        high = float(np.percentile(flat, upper_percentile))
        # Add the cell index, lower and upper percentiles to the output list
        results.append((i, low, high))
    # Return the results list
    return results


def _process_cell_box_for_percentiles(args):
    """Worker: read one cell's bbox, compute percentiles.
    Returns (cell_idx, low, high)."""
    (image_path, channel_index, y_axis, x_axis, c_axis, cell_idx, x, y, w, h,
     lower_percentile, upper_percentile) = args
    if w <= 0 or h <= 0:
        return (cell_idx, np.nan, np.nan)
    try:
        # Read the cell bbox region from the image channel
        box = _read_channel_from_ometiff_zarr(
            image_path, channel_index, y_axis, x_axis, c_axis,
            y_lo=y, y_hi=y + h, x_lo=x, x_hi=x + w,
        )
        # Flatten the array of channel signal in the cell bbox
        flat = np.asarray(box).ravel()
        # Skip if no pixels in the cell bbox
        if flat.size == 0:
            return (cell_idx, np.nan, np.nan)
        # Compute the lower and upper percentiles of the channel signal in the cell bbox
        low = float(np.percentile(flat, lower_percentile))
        high = float(np.percentile(flat, upper_percentile))
        # Return the cell index, lower and upper percentiles
        return (cell_idx, low, high)
    except Exception:
        return (cell_idx, np.nan, np.nan)


def compute_normalization_from_bounds(
    image_path,
    measurements_tsv_path,
    channel_index,
    lower_percentile=10.0,
    upper_percentile=90.0,
    tile_size=5000,
):
    """
    Compute min/max for normalization for entire preprocess field from per-cell
    upper/lower percentiles
    by tile-based reading of bounding boxes from the TSV and the image.
    Returns (global_min, global_max) for use with preprocess_roi.
    """
    # Ensure image file path exists and is valid
    image_path = os.path.abspath(os.path.normpath(image_path.strip()))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path!r}")

    # Read the measurements TSV file
    df = pd.read_csv(measurements_tsv_path, sep="\t")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    # Rename columns to standardize QuPath export names
    _COLUMN_ALIASES = {
        "Bounds x": "Bounds_x",
        "Bounds y": "Bounds_y",
        "Bounds width": "Bounds_width",
        "Bounds height": "Bounds_height",
        "Object  ID": "Object ID",
    }
    rename = {k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    # Ensure the required columns are present
    required_cols = ["Object ID", "Bounds_x", "Bounds_y", "Bounds_width", "Bounds_height"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Measurements TSV missing column(s): {missing}. Actual: {list(df.columns)}")

    # Number of cells
    n_cells = len(df)
    # Get the x, y, width, height of the cells' bounding boxes
    x_col = np.floor(df["Bounds_x"].astype(float)).astype(int)
    y_col = np.floor(df["Bounds_y"].astype(float)).astype(int)
    w_col = np.ceil(df["Bounds_width"].astype(float)).astype(int)
    h_col = np.ceil(df["Bounds_height"].astype(float)).astype(int)

    # Get the image dimensions and axes
    shape, y_axis, x_axis, c_axis = _ometiff_shape_and_axes(image_path)
    height = int(shape[y_axis])
    width = int(shape[x_axis])
    # Calculate the step size for the x and y dimensions
    step_x = max(1, tile_size)
    step_y = max(1, tile_size)
    # Calculate the number of tiles in the x and y dimensions
    n_tx = max(1, int(np.ceil(width / step_x)))
    n_ty = max(1, int(np.ceil(height / step_y)))

    # Empty nan arrays
    cell_lows = np.full(n_cells, np.nan)
    cell_highs = np.full(n_cells, np.nan)

    # Build list of cells in tile -> list of (cell_idx, x, y, w, h)
    tile_cells = {}
    for i in range(n_cells):
        x, y = int(x_col.iloc[i]), int(y_col.iloc[i])
        w, h = int(w_col.iloc[i]), int(h_col.iloc[i])
        # Cell centroid
        cx, cy = x + w // 2, y + h // 2
        # Tile index which contains cell centroid
        ty = min(int(cy // step_y), n_ty - 1)
        tx = min(int(cx // step_x), n_tx - 1)
        # Bounds of that tile
        x_lo = tx * step_x
        x_hi = min(tx * step_x + tile_size, width)
        y_lo = ty * step_y
        y_hi = min(ty * step_y + tile_size, height)
        # If the cell is fully in the tile, add it to the tile cells dictionary
        if _tile_contains_box(x_lo, y_lo, x_hi, y_hi, x, y, x + w, y + h):
            key = (ty, tx)
            if key not in tile_cells:
                tile_cells[key] = []
            tile_cells[key].append((i, x, y, w, h))

    # Process tiles in parallel with multiprocessing
    # Each worker reads one tile and computes percentiles for its cells
    tile_tasks = []
    # Iterate over all the tiles
    for ty in range(n_ty):
        for tx in range(n_tx):
            key = (ty, tx)
            # If the tile is not in the dictionary or is empty, skip it because it doesn't contain any cells
            if key not in tile_cells or not tile_cells[key]:
                continue
            # Calculate the bounds of the tile
            y_lo = ty * step_y
            y_hi = min(ty * step_y + tile_size, height)
            x_lo = tx * step_x
            x_hi = min(tx * step_x + tile_size, width)
            # Add the tile to the tile tasks list
            tile_tasks.append((
                image_path, channel_index, y_axis, x_axis, c_axis,
                y_lo, y_hi, x_lo, x_hi,
                tile_cells[key],
                lower_percentile, upper_percentile,
            ))
    # Number of workers to use for the multiprocessing
    n_workers = min(max(1, len(tile_tasks)), mp.cpu_count() or 8)
    if tile_tasks:
        # Create a pool of workers to process the tiles
        with mp.Pool(processes=n_workers) as pool:
            for result_list in pool.map(_process_tile_for_percentiles, tile_tasks):
                # Record the lower and upper percentiles for each cell
                for (i, low, high) in result_list:
                    cell_lows[i] = low
                    cell_highs[i] = high

    # Cells not fully in any tile: read each box in parallel
    # Works because we did not use overlapping tiles so there will be some cells that are not fully in any tile
    fallback_indices = [i for i in range(n_cells) if np.isnan(cell_lows[i])]
    if fallback_indices:
        # Create a list of tasks to process the cells that are not fully in any tile
        fallback_tasks = [
            (
                image_path, channel_index, y_axis, x_axis, c_axis,
                i, int(x_col.iloc[i]), int(y_col.iloc[i]), int(w_col.iloc[i]), int(h_col.iloc[i]),
                lower_percentile, upper_percentile,
            )
            for i in fallback_indices
        ]
        # Number of workers to use for the multiprocessing
        n_workers_fb = min(max(1, len(fallback_tasks)), mp.cpu_count() or 8)
        with mp.Pool(processes=n_workers_fb) as pool:
            # Process the cells that are not fully in any tile
            for (i, low, high) in pool.map(_process_cell_box_for_percentiles, fallback_tasks):
                if np.isfinite(low) and np.isfinite(high):
                    # Record the lower and upper percentiles for the cell
                    cell_lows[i] = low
                    cell_highs[i] = high

    valid = np.isfinite(cell_lows) & np.isfinite(cell_highs)
    if not np.any(valid):
        raise ValueError("Could not compute any per-cell percentiles for normalization.")
    # Compute the global minimum and maximum of the lower and upper percentiles
    global_min = float(np.nanmin(cell_lows[valid]))
    global_max = float(np.nanmax(cell_highs[valid]))
    _log(f"[PhenoBIC] Normalization from bounds: min={global_min:.2f}, max={global_max:.2f} (channel {channel_index}, {np.sum(valid)} cells)")
    return global_min, global_max


def run_single_image(
    image_path,
    measurements_tsv_path,
    min_int,
    max_int,
    channel_index,
    buffer_ratio,
    output_csv_path,
    model_path,
    num_cells_batch=4000,
    tile_size=10000,
    output_min_json=None,
    output_max_json=None,
    channel_name=None,
    lower_percentile=10.0,
    upper_percentile=90.0,
):
    """
    Run PhenoBIC phenotype inference for one image using tile-based reads.
    Splits the image into overlapping tiles, assigns each cell to a tile, and processes tile by tile.
    """
    # Ensure image file path exists and is valid
    image_path = os.path.abspath(os.path.normpath(image_path.strip()))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path!r}")

    # Compute normalization min/max from image and bounds if not provided
    if min_int is None or max_int is None:
        _log(f"[PhenoBIC] Computing normalization percentiles from bounds (channel {channel_index})...")
        min_int, max_int = compute_normalization_from_bounds(
            image_path, measurements_tsv_path, channel_index,
            lower_percentile=lower_percentile, upper_percentile=upper_percentile, tile_size=min(5000, tile_size),
        )

    # Save min/max to JSON files (merge with existing so multi-channel runs accumulate)
    if channel_name and output_min_json:
        _write_normalization_json(output_min_json, channel_name, min_int)
        _log(f"[PhenoBIC] Wrote min normalization: {output_min_json}")
    if channel_name and output_max_json:
        _write_normalization_json(output_max_json, channel_name, max_int)
        _log(f"[PhenoBIC] Wrote max normalization: {output_max_json}")

    # Load the PhenoBIC model
    _log(f"[PhenoBIC] Loading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(os.path.abspath(model_path.strip()), compile=False)

    # Load and normalize measurements TSV (QuPath export).
    df = pd.read_csv(measurements_tsv_path, sep="\t")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    # Rename columns to standardize QuPath export names
    _COLUMN_ALIASES = {
        "Bounds x": "Bounds_x",
        "Bounds y": "Bounds_y",
        "Bounds width": "Bounds_width",
        "Bounds height": "Bounds_height",
        "Object  ID": "Object ID",
    }
    rename = {k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(columns=rename)
    # Ensure the required columns are present
    required_cols = ["Object ID", "Bounds_x", "Bounds_y", "Bounds_width", "Bounds_height"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Measurements TSV missing column(s): {missing}. Actual: {list(df.columns)}")
    
    # Number of cells to process
    num_cells = len(df)
    # Get the image name
    image_name = os.path.basename(image_path)
    # Get the x, y, width, height, and buffer of the cell bounding boxes
    x_col = np.floor(df["Bounds_x"].astype(float)).astype(int)
    y_col = np.floor(df["Bounds_y"].astype(float)).astype(int)
    w_col = np.ceil(df["Bounds_width"].astype(float)).astype(int)
    h_col = np.ceil(df["Bounds_height"].astype(float)).astype(int)
    x_buf_col = np.ceil(w_col * buffer_ratio).astype(int)
    y_buf_col = np.ceil(h_col * buffer_ratio).astype(int)
    object_ids = df["Object ID"].values

    # Get the image dimensions and axes (no pixel load).
    shape, y_axis, x_axis, c_axis = _ometiff_shape_and_axes(image_path)
    height = int(shape[y_axis])
    width = int(shape[x_axis])

    # Tile grid: overlap so each cell's buffered box fits in at least one tile.
    # Create an ImageDataGenerator for preprocessing the images
    gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    # Calculate the buffered width and height of the cells
    buffered_w = w_col + 2 * x_buf_col
    buffered_h = h_col + 2 * y_buf_col
    # Calculate the overlap between tiles (max of buffered width and height of cells)
    overlap = int(min(tile_size - 1, max(np.max(buffered_w), np.max(buffered_h)))) if num_cells else 0
    # Calculate the step size for the x and y dimensions
    step_x = max(1, tile_size - overlap)
    step_y = max(1, tile_size - overlap)
    # Calculate the number of tiles in the x and y dimensions
    n_tx = max(1, int(np.ceil(width / step_x)))
    n_ty = max(1, int(np.ceil(height / step_y)))
    _log(f"[PhenoBIC] Image: {image_name} — {num_cells} cells, tile {tile_size} px, overlap {overlap} px, {n_ty}x{n_tx} tiles")

    # Assign each cell to a tile (key = (ty, tx)); value = list of (object_id, x, y, w, h, x_buf, y_buf).
    tile_cells = {}
    # iterate over all the cells
    for i in range(num_cells):
        # Get the x, y, width, height, and buffer of the cell bounding boxes
        x, y = int(x_col.iloc[i]), int(y_col.iloc[i])
        w, h = int(w_col.iloc[i]), int(h_col.iloc[i])
        x_buf, y_buf = int(x_buf_col.iloc[i]), int(y_buf_col.iloc[i])
        bx_lo, by_lo = x - x_buf, y - y_buf
        bx_hi, by_hi = x + w + x_buf, y + h + y_buf
        # Calculate the centroid of the cell
        cx, cy = x + w // 2, y + h // 2
        # Assign the cell to a tile
        ty, tx = _assign_cell_to_tile(cx, cy, bx_lo, by_lo, bx_hi, by_hi, n_ty, n_tx, step_y, step_x, tile_size, height, width)
        # Create a dictionary of tiles and the cells in them
        key = (ty, tx)
        # If the tile is not in the dictionary, create a entry for it
        if key not in tile_cells:
            tile_cells[key] = []
        # Add the cell to the dictionary
        tile_cells[key].append((object_ids[i], x, y, w, h, x_buf, y_buf))

    # Process each tile: load tile, extract ROIs, preprocess, predict, store by object ID.
    # Dictionary to store Cell object IDs and the prediction for each cell
    predictions_dict = {}
    # Iterate over all the tiles
    for ty in range(n_ty):
        for tx in range(n_tx):
            key = (ty, tx)
            # If the tile is not in the dictionary or is empty, skip it because it doesn't contain any cells
            if key not in tile_cells or not tile_cells[key]:
                continue
            cell_list = tile_cells[key]
            # Calculate the lower and upper bounds of the tile
            y_lo = ty * step_y
            y_hi = min(ty * step_y + tile_size, height)
            x_lo = tx * step_x
            x_hi = min(tx * step_x + tile_size, width)
            # Calculate the bounds of the cells in the tile relative to the tile
            bounds_tile_rel = [(x - x_lo, y - y_lo, w, h, x_buf, y_buf) for (_, x, y, w, h, x_buf, y_buf) in cell_list]
            # Get the object IDs of the cells in the tile
            oids = [oid for (oid, *_) in cell_list]

            # Create a pool of workers to extract the ROIs from the cells (multiprocessing)
            with mp.Pool(
                # Initialize the worker with the image path, channel index, and the bounds of the tile
                initializer=_init_worker_tile,
                initargs=(image_path, channel_index, y_lo, y_hi, x_lo, x_hi, y_axis, x_axis, c_axis),
            ) as pool:
                # Iterate over the cells in the tile in batches
                for start in range(0, len(bounds_tile_rel), num_cells_batch):
                    # Calculate the end index of the batch
                    end = min(start + num_cells_batch, len(bounds_tile_rel))
                    # Get the bounds of the cells in the batch
                    batch_bounds = bounds_tile_rel[start:end]
                    # Get the object IDs of the cells in the batch
                    batch_oids = oids[start:end]
                    # Extract the ROIs from the cells
                    rois = pool.map(_extract_roi, batch_bounds)
                    # Preprocess the ROIs
                    rois = [preprocess_roi(r, min_int, max_int) for r in rois]
                    # Resize the ROIs to 48x48 pixels
                    rois = np.array([
                        np.asarray(Image.fromarray(np.asarray(im, dtype=np.uint8)).resize((48, 48), Image.NEAREST))
                        for im in rois
                    ])
                    # PhenoBIC prediction for each cell in the batch
                    flow = gen.flow(x=rois, batch_size=32, shuffle=False)
                    pred = model.predict(flow)
                    pred = pred.reshape(-1)
                    # Convert the predictions to binary cell expressionlabels
                    labels = np.where(pred <= 0.5, "neg", "pos")
                    # Store the predictions for each cell
                    for j, oid in enumerate(batch_oids):
                        predictions_dict[oid] = labels[j]
            _log(f"[PhenoBIC] Tile ({ty + 1},{tx + 1})/{n_ty}x{n_tx} — {len(cell_list)} cells done ({len(predictions_dict)}/{num_cells})")

    # Convert predictions dictionary to a list of predictions
    predictions_list = [predictions_dict[oid] for oid in object_ids]
    _log(f"[PhenoBIC] Writing results: {os.path.basename(output_csv_path)}")
    # Create a DataFrame with the object IDs and the predictions
    out = pd.DataFrame({"Object ID": df["Object ID"], "Class": predictions_list})
    # Write the DataFrame to a CSV file
    out.to_csv(output_csv_path, index=False)
    _log(f"[PhenoBIC] Done. {num_cells} cells classified.")
    # Return the path to the output CSV file
    return output_csv_path

# Parse command line arguments
def parse_args():
    p = argparse.ArgumentParser(description="PhenoBIC phenotype inference (tile-based, single image).")
    p.add_argument("--image", type=str, help="Path to OME-TIFF (or compatible) image")
    p.add_argument("--measurements-tsv", type=str, help="Path to measurements TSV (Object ID, Bounds_*, etc.)")
    p.add_argument("--min", type=float, default=None, help="Lower intensity for normalization clip (default: compute from bounds)")
    p.add_argument("--max", type=float, default=None, help="Upper intensity for normalization clip (default: compute from bounds)")
    p.add_argument("--lower-percentile", type=float, default=10.0, help="Percentile for lower clip when computing from bounds")
    p.add_argument("--upper-percentile", type=float, default=90.0, help="Percentile for upper clip when computing from bounds")
    p.add_argument("--channel-index", type=int, help="0-based channel index")
    p.add_argument("--buffer-ratio", type=float, default=0.1, help="Cell bounding box buffer ratio")
    p.add_argument("--output-csv", type=str, help="Output CSV path")
    p.add_argument("--model-path", type=str, help="Path to .keras PhenoBIC model")
    p.add_argument("--num-cells-batch", type=int, default=4000, help="Batch size for cell ROI extraction per tile")
    p.add_argument("--tile-size", type=int, default=10000, help="Tile size in pixels (tile_size x tile_size)")
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU (CPU only)")
    p.add_argument("--output-min-json", type=str, default=None, help="Path to JSON file to write/merge min normalization chanelwise params")
    p.add_argument("--output-max-json", type=str, default=None, help="Path to JSON file to write/merge max normalization chanelwise params")
    p.add_argument("--channel-name", type=str, default=None, help="Channel name for JSON key when saving normalization")
    return p.parse_args()

# Main function to run the script
if __name__ == "__main__":
    args = parse_args()
    # Check if the measurements TSV file is provided
    if not args.measurements_tsv:
        # Print an error message and exit
        print("Missing --measurements-tsv. Use --help.", file=sys.stderr)
        sys.exit(1)
    # Check if all the required arguments are provided
    required = (args.image, args.channel_index, args.output_csv, args.model_path, args.num_cells_batch)
    if all(r is not None for r in required):
        # Run the script
        run_single_image(
            args.image,
            args.measurements_tsv,
            args.min,
            args.max,
            args.channel_index,
            args.buffer_ratio,
            args.output_csv,
            args.model_path,
            args.num_cells_batch,
            args.tile_size,
            output_min_json=args.output_min_json,
            output_max_json=args.output_max_json,
            channel_name=args.channel_name,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
        )
        sys.exit(0)
    print("Missing required arguments. Use --help for usage.", file=sys.stderr)
    sys.exit(1)
