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
import sys

# Disable GPU before TensorFlow is imported (e.g. --no-gpu from QuPath/Groovy).
if "--no-gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import tifffile
import zarr



# OME-TIFF axis and series selection

def _ometiff_zarr_axes(shape):
    """Infer (y_axis, x_axis, c_axis) from array shape. c_axis is None for 2D."""
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
    if axes and "Y" in axes and "X" in axes:
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
    if not tif.series:
        raise ValueError("No series in TIFF")
    best_i = 0
    best_size = 0
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
        idx = _series_index_with_largest_yx(tif)
        series = tif.series[idx]
        shape = series.shape
        axes = getattr(series, "axes", None)
    y_axis, x_axis, c_axis = _axes_from_series_axes(axes, shape)
    return shape, y_axis, x_axis, c_axis


# Tile-based channel reading (tifffile + zarr)

def _slice_for_axis(i, y_axis, x_axis, c_axis, channel_index, y_lo, y_hi, x_lo, x_hi):
    """Return the slice for axis i: Y/X window or channel index as needed."""
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
    with tifffile.TiffFile(image_path) as tif:
        idx = _series_index_with_largest_yx(tif)
        store = tif.series[idx].aszarr()
        root = zarr.open(store, mode="r")

        # Single array vs Group (pyramid): pick array with largest X*Y.
        if hasattr(root, "ndim"):
            arr = root
        else:
            keys = list(getattr(root, "array_keys", lambda: list(root.keys()))())
            if not keys:
                raise ValueError("No array in OME-TIFF zarr store")
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

        nd = arr.ndim
        idx = tuple(
            _slice_for_axis(i, y_axis, x_axis, c_axis, channel_index, y_lo, y_hi, x_lo, x_hi)
            for i in range(nd)
        )
        out = np.asarray(arr[idx], dtype=np.float64)
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
    """Crop the worker's channel plane for one cell. Bounds are (x, y, w, h, x_buf, y_buf) in tile-relative coords."""
    global _worker_channel_plane, _worker_y_axis, _worker_x_axis
    plane = _worker_channel_plane
    ya, xa = _worker_y_axis, _worker_x_axis
    x, y, w, h, x_buf, y_buf = bounds
    y_lo = max(0, int(y - y_buf))
    y_hi = min(plane.shape[ya], int(y + h + y_buf))
    x_lo = max(0, int(x - x_buf))
    x_hi = min(plane.shape[xa], int(x + w + x_buf))
    s = [slice(None), slice(None)]
    s[ya] = slice(y_lo, y_hi)
    s[xa] = slice(x_lo, x_hi)
    return np.asarray(plane[tuple(s)])


def preprocess_roi(roi, min_val, max_val):
    """Linearly scale ROI to [0,255] and replicate to 3 channels for the model."""
    scaled = np.clip((roi.astype(np.float64) - min_val) / (max_val - min_val), 0, 1)
    return np.stack((np.uint8(scaled * 255),) * 3, axis=-1)


# Tile assignment and main pipeline

def _log(msg):
    """Print progress (e.g. for QuPath log)."""
    print(msg, flush=True)


def _tile_contains_box(x_lo, y_lo, x_hi, y_hi, bx_lo, by_lo, bx_hi, by_hi):
    """True if tile [x_lo,x_hi) x [y_lo,y_hi) fully contains the box [bx_lo,bx_hi] x [by_lo,by_hi]."""
    return x_lo <= bx_lo and bx_hi <= x_hi and y_lo <= by_lo and by_hi <= y_hi


def _assign_cell_to_tile(cx, cy, bx_lo, by_lo, bx_hi, by_hi, n_ty, n_tx, step_y, step_x, tile_size, height, width):
    """Return (ty, tx) of the tile that contains the cell centroid and its full buffered box; else (0, 0)."""
    for ty in range(n_ty):
        for tx in range(n_tx):
            y_lo = ty * step_y
            y_hi = min(ty * step_y + tile_size, height)
            x_lo = tx * step_x
            x_hi = min(tx * step_x + tile_size, width)
            if x_lo <= cx < x_hi and y_lo <= cy < y_hi and _tile_contains_box(x_lo, y_lo, x_hi, y_hi, bx_lo, by_lo, bx_hi, by_hi):
                return (ty, tx)
    return (0, 0)


def run_single_image(
    image_path,
    measurements_tsv_path,
    min_int,
    max_int,
    channel_index,
    buffer_ratio,
    output_csv_path,
    model_path,
    num_cells_batch=2000,
    tile_size=10000,
):
    """
    Run PhenoBIC phenotype inference for one image using tile-based reads only.
    Splits the image into overlapping tiles, assigns each cell to a tile, and processes tile by tile.
    """
    image_path = os.path.abspath(os.path.normpath(image_path.strip()))
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path!r}")

    _log(f"[PhenoBIC] Loading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(os.path.abspath(model_path.strip()), compile=False)

    # Load and normalize measurements TSV (QuPath export).
    df = pd.read_csv(measurements_tsv_path, sep="\t")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
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
    required_cols = ["Object ID", "Bounds_x", "Bounds_y", "Bounds_width", "Bounds_height"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Measurements TSV missing column(s): {missing}. Actual: {list(df.columns)}")

    num_cells = len(df)
    image_name = os.path.basename(image_path)
    x_col = np.floor(df["Bounds_x"].astype(float)).astype(int)
    y_col = np.floor(df["Bounds_y"].astype(float)).astype(int)
    w_col = np.ceil(df["Bounds_width"].astype(float)).astype(int)
    h_col = np.ceil(df["Bounds_height"].astype(float)).astype(int)
    x_buf_col = np.ceil(w_col * buffer_ratio).astype(int)
    y_buf_col = np.ceil(h_col * buffer_ratio).astype(int)
    object_ids = df["Object ID"].values

    # Image dimensions and axes (no pixel load).
    shape, y_axis, x_axis, c_axis = _ometiff_shape_and_axes(image_path)
    height = int(shape[y_axis])
    width = int(shape[x_axis])

    # Tile grid: overlap so each cell's buffered box fits in at least one tile.
    gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    buffered_w = w_col + 2 * x_buf_col
    buffered_h = h_col + 2 * y_buf_col
    overlap = int(min(tile_size - 1, max(np.max(buffered_w), np.max(buffered_h)))) if num_cells else 0
    step_x = max(1, tile_size - overlap)
    step_y = max(1, tile_size - overlap)
    n_tx = max(1, int(np.ceil(width / step_x)))
    n_ty = max(1, int(np.ceil(height / step_y)))
    _log(f"[PhenoBIC] Image: {image_name} — {num_cells} cells, tile {tile_size} px, overlap {overlap} px, {n_ty}x{n_tx} tiles")

    # Assign each cell to a tile (key = (ty, tx)); value = list of (object_id, x, y, w, h, x_buf, y_buf).
    tile_cells = {}
    for i in range(num_cells):
        x, y = int(x_col.iloc[i]), int(y_col.iloc[i])
        w, h = int(w_col.iloc[i]), int(h_col.iloc[i])
        x_buf, y_buf = int(x_buf_col.iloc[i]), int(y_buf_col.iloc[i])
        bx_lo, by_lo = x - x_buf, y - y_buf
        bx_hi, by_hi = x + w + x_buf, y + h + y_buf
        cx, cy = x + w // 2, y + h // 2
        ty, tx = _assign_cell_to_tile(cx, cy, bx_lo, by_lo, bx_hi, by_hi, n_ty, n_tx, step_y, step_x, tile_size, height, width)
        key = (ty, tx)
        if key not in tile_cells:
            tile_cells[key] = []
        tile_cells[key].append((object_ids[i], x, y, w, h, x_buf, y_buf))

    # Process each tile: load tile, extract ROIs, preprocess, predict, store by object ID.
    predictions_dict = {}
    for ty in range(n_ty):
        for tx in range(n_tx):
            key = (ty, tx)
            if key not in tile_cells or not tile_cells[key]:
                continue
            cell_list = tile_cells[key]
            y_lo = ty * step_y
            y_hi = min(ty * step_y + tile_size, height)
            x_lo = tx * step_x
            x_hi = min(tx * step_x + tile_size, width)
            bounds_tile_rel = [(x - x_lo, y - y_lo, w, h, x_buf, y_buf) for (_, x, y, w, h, x_buf, y_buf) in cell_list]
            oids = [oid for (oid, *_) in cell_list]

            with mp.Pool(
                initializer=_init_worker_tile,
                initargs=(image_path, channel_index, y_lo, y_hi, x_lo, x_hi, y_axis, x_axis, c_axis),
            ) as pool:
                for start in range(0, len(bounds_tile_rel), num_cells_batch):
                    end = min(start + num_cells_batch, len(bounds_tile_rel))
                    batch_bounds = bounds_tile_rel[start:end]
                    batch_oids = oids[start:end]
                    rois = pool.map(_extract_roi, batch_bounds)
                    rois = [preprocess_roi(r, min_int, max_int) for r in rois]
                    rois = np.array([
                        np.asarray(Image.fromarray(np.asarray(im, dtype=np.uint8)).resize((48, 48), Image.NEAREST))
                        for im in rois
                    ])
                    flow = gen.flow(x=rois, batch_size=32, shuffle=False)
                    pred = model.predict(flow)
                    pred = pred.reshape(-1)
                    labels = np.where(pred <= 0.5, "neg", "pos")
                    for j, oid in enumerate(batch_oids):
                        predictions_dict[oid] = labels[j]
            _log(f"[PhenoBIC] Tile ({ty + 1},{tx + 1})/{n_ty}x{n_tx} — {len(cell_list)} cells done ({len(predictions_dict)}/{num_cells})")

    predictions_list = [predictions_dict[oid] for oid in object_ids]
    _log(f"[PhenoBIC] Writing results: {os.path.basename(output_csv_path)}")
    out = pd.DataFrame({"Object ID": df["Object ID"], "Class": predictions_list})
    out.to_csv(output_csv_path, index=False)
    _log(f"[PhenoBIC] Done. {num_cells} cells classified.")
    return output_csv_path


def parse_args():
    p = argparse.ArgumentParser(description="PhenoBIC phenotype inference (tile-based, single image).")
    p.add_argument("--image", type=str, help="Path to OME-TIFF (or compatible) image")
    p.add_argument("--measurements-tsv", type=str, help="Path to measurements TSV (Object ID, Bounds_*, etc.)")
    p.add_argument("--min", type=float, help="Lower intensity for normalization clip")
    p.add_argument("--max", type=float, help="Upper intensity for normalization clip")
    p.add_argument("--channel-index", type=int, help="0-based channel index")
    p.add_argument("--buffer-ratio", type=float, default=0.1, help="Cell bounding box buffer ratio")
    p.add_argument("--output-csv", type=str, help="Output CSV path")
    p.add_argument("--model-path", type=str, help="Path to .keras PhenoBIC model")
    p.add_argument("--num-cells-batch", type=int, default=2000, help="Batch size for cell ROI extraction per tile")
    p.add_argument("--tile-size", type=int, default=10000, help="Tile size in pixels (tile_size x tile_size)")
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU (CPU only)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.measurements_tsv:
        print("Missing --measurements-tsv. Use --help.", file=sys.stderr)
        sys.exit(1)
    required = (args.image, args.min, args.max, args.channel_index, args.output_csv, args.model_path, args.num_cells_batch)
    if all(r is not None for r in required):
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
        )
        sys.exit(0)
    print("Missing required arguments. Use --help for usage.", file=sys.stderr)
    sys.exit(1)
