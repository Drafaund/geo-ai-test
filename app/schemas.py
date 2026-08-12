from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(description="Easting/X in the raster's native CRS")
    y: float = Field(description="Northing/Y in the raster's native CRS")
    lon: float = Field(description="Longitude in WGS84 (EPSG:4326)")
    lat: float = Field(description="Latitude in WGS84 (EPSG:4326)")


class DetectionResponse(BaseModel):
    filename: str
    crs: str | None
    tile_width: int
    tile_height: int
    conf_threshold: float
    nms_threshold: float
    min_distance: float
    total_detections: int
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
