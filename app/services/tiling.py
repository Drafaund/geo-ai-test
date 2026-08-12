"""Split a georeferenced raster into fixed-size, overlapping tiles.

Uses rasterio windowed, boundless reads so tiles are always exactly
``tile_width`` x ``tile_height`` (edge tiles are zero-padded automatically)
and each tile keeps a correct georeferencing transform, including for
rasters smaller than a single tile.
"""

import math
from pathlib import Path

import rasterio
from rasterio.windows import Window, transform as window_transform


def _tile_offsets(img_size: int, tile_size: int) -> list[int]:
    """Compute the top-left pixel offset of each tile along one axis."""
    num_tiles = math.ceil(img_size / tile_size)
    if num_tiles <= 1:
        return [0]

    step = (img_size - tile_size) / (num_tiles - 1)
    offsets = [round(i * step) for i in range(num_tiles)]
    offsets[-1] = img_size - tile_size
    return offsets


def tile_raster(input_path: Path, output_dir: Path, tile_width: int, tile_height: int) -> list[Path]:
    """Tile a raster and write each tile as a georeferenced GeoTIFF.

    Returns the list of written tile paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: list[Path] = []

    with rasterio.open(input_path) as src:
        x_offsets = _tile_offsets(src.width, tile_width)
        y_offsets = _tile_offsets(src.height, tile_height)

        count = 0
        for left in x_offsets:
            for upper in y_offsets:
                window = Window(left, upper, tile_width, tile_height)
                tile_data = src.read(window=window, boundless=True, fill_value=0)
                tile_tr = window_transform(window, src.transform)

                tile_path = output_dir / f"tile_{count}.tif"
                with rasterio.open(
                    tile_path,
                    "w",
                    driver="GTiff",
                    height=tile_height,
                    width=tile_width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=tile_tr,
                ) as dst:
                    dst.write(tile_data)

                tile_paths.append(tile_path)
                count += 1

    return tile_paths
