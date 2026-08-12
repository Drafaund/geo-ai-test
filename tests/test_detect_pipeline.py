import io
import zipfile

import pytest

from tests.conftest import SAMPLE_RASTER

pytestmark = pytest.mark.skipif(not SAMPLE_RASTER.exists(), reason="sawit.tif sample raster not present")


def test_detect_json_end_to_end(client):
    with SAMPLE_RASTER.open("rb") as f:
        resp = client.post("/api/v1/detect", files={"file": ("sawit.tif", f, "image/tiff")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["crs"] == "EPSG:32749"
    assert body["total_detections"] > 0
    assert len(body["detections"]) == body["total_detections"]

    det = body["detections"][0]
    assert det["class_name"] == "sawit"
    assert 0.0 <= det["confidence"] <= 1.0
    # sanity check: reprojected coordinates fall within Indonesia's bounding box
    assert 90 <= det["lon"] <= 141
    assert -11 <= det["lat"] <= 6


def test_detect_shapefile_end_to_end(client):
    with SAMPLE_RASTER.open("rb") as f:
        resp = client.post("/api/v1/detect?output=shapefile", files={"file": ("sawit.tif", f, "image/tiff")})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    names = set(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())
    assert {"sawit.shp", "sawit.shx", "sawit.dbf", "sawit.prj"} <= names
