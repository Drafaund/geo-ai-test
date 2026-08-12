"""End-to-end orchestration: tile a raster, detect, and geo-reference results."""

import shutil
import tempfile
from pathlib import Path

import cv2
import geopandas as gpd
import rasterio
from shapely.geometry import Point

from app.services.detection import PalmDetector, get_gsd, merge_nearby_detections
from app.services.tiling import tile_raster

DETECTION_COLUMNS = ["class_name", "confidence", "geometry"]


def run_detection_pipeline(
    detector: PalmDetector,
    raster_path: Path,
    tile_width: int,
    tile_height: int,
    conf_threshold: float,
    nms_threshold: float,
    min_distance: float,
) -> gpd.GeoDataFrame:
    """Tile a raster, run detection on every tile, and return deduplicated
    detections as a GeoDataFrame in the raster's native CRS."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="palm_tiles_"))
    try:
        tile_paths = tile_raster(raster_path, tmp_dir, tile_width, tile_height)
        gsd_x, gsd_y = get_gsd(raster_path)

        with rasterio.open(raster_path) as src:
            crs = src.crs

        records = []
        for tile_path in tile_paths:
            tile_img = cv2.imread(str(tile_path))
            if tile_img is None:
                continue

            with rasterio.open(tile_path) as tile_src:
                bounds = tile_src.bounds

            detections = detector.detect(tile_img, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
            for det in detections:
                geo_x = (det["x"] + det["width"] / 2) * gsd_x + bounds.left
                geo_y = bounds.top - (det["y"] + det["height"] / 2) * gsd_y
                records.append(
                    {
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "geometry": Point(geo_x, geo_y),
                    }
                )

        if not records:
            return gpd.GeoDataFrame(columns=DETECTION_COLUMNS, geometry="geometry", crs=crs)

        gdf = gpd.GeoDataFrame(records, crs=crs)
        return merge_nearby_detections(gdf, min_distance)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
