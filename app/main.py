"""FastAPI service for oil-palm tree detection from georeferenced orthophotos."""

import io
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import rasterio
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from rasterio.errors import RasterioIOError

from app.config import settings
from app.schemas import Detection, DetectionResponse, HealthResponse
from app.services.detection import PalmDetector
from app.services.pipeline import run_detection_pipeline

detector: PalmDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = PalmDetector(settings.model_path, settings.labels_path)
    yield
    detector = None


app = FastAPI(title="Palm Counting API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=detector is not None)


@app.post("/api/v1/detect")
async def detect(
    file: UploadFile = File(..., description="Georeferenced orthophoto (GeoTIFF)"),
    tile_width: int = Query(default=None, gt=0),
    tile_height: int = Query(default=None, gt=0),
    conf_threshold: float = Query(default=None, ge=0.0, le=1.0),
    nms_threshold: float = Query(default=None, ge=0.0, le=1.0),
    min_distance: float = Query(default=None, ge=0.0),
    output: Literal["json", "shapefile"] = Query(default="json"),
):
    """Detect palm trees in an uploaded GeoTIFF and return their coordinates."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="File must be a .tif/.tiff GeoTIFF")

    tile_width = tile_width or settings.tile_width
    tile_height = tile_height or settings.tile_height
    conf_threshold = conf_threshold if conf_threshold is not None else settings.conf_threshold
    nms_threshold = nms_threshold if nms_threshold is not None else settings.nms_threshold
    min_distance = min_distance if min_distance is not None else settings.min_distance

    work_dir = Path(tempfile.mkdtemp(prefix="palm_upload_"))
    try:
        raster_path = work_dir / file.filename
        with raster_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            with rasterio.open(raster_path):
                pass
        except RasterioIOError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid or unreadable raster: {exc}") from exc

        gdf = run_detection_pipeline(
            detector=detector,
            raster_path=raster_path,
            tile_width=tile_width,
            tile_height=tile_height,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            min_distance=min_distance,
        )

        if output == "shapefile":
            return _shapefile_response(gdf, work_dir, file.filename)

        return _json_response(
            gdf, file.filename, tile_width, tile_height, conf_threshold, nms_threshold, min_distance
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _json_response(
    gdf, filename, tile_width, tile_height, conf_threshold, nms_threshold, min_distance
) -> DetectionResponse:
    crs = gdf.crs.to_string() if gdf.crs else None
    wgs84_geoms = gdf.to_crs(4326).geometry if len(gdf) and gdf.crs else gdf.geometry

    detections = [
        Detection(
            class_name=row.class_name,
            confidence=row.confidence,
            x=row.geometry.x,
            y=row.geometry.y,
            lon=wgs_point.x,
            lat=wgs_point.y,
        )
        for row, wgs_point in zip(gdf.itertuples(), wgs84_geoms, strict=False)
    ]

    return DetectionResponse(
        filename=filename,
        crs=crs,
        tile_width=tile_width,
        tile_height=tile_height,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        min_distance=min_distance,
        total_detections=len(detections),
        detections=detections,
    )


def _shapefile_response(gdf, work_dir: Path, source_filename: str) -> StreamingResponse:
    stem = Path(source_filename).stem
    shp_dir = work_dir / "shapefile"
    shp_dir.mkdir(exist_ok=True)
    shp_path = shp_dir / f"{stem}.shp"
    gdf.to_file(shp_path)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for part in shp_dir.iterdir():
            zf.write(part, arcname=part.name)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}.zip"'},
    )
