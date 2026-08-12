"""YOLO (ONNX, OpenCV DNN) inference and post-processing helpers."""

import threading
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import yaml
from shapely.ops import unary_union
from shapely.strtree import STRtree


class PalmDetector:
    """Loads the ONNX model once and runs thread-safe inference on tiles."""

    def __init__(self, model_path: Path, labels_path: Path):
        self.labels = self._load_labels(labels_path)
        self.net = self._load_model(model_path)
        self._lock = threading.Lock()  # cv2.dnn.Net is not safe for concurrent forward() calls

    @staticmethod
    def _load_labels(labels_path: Path) -> dict[int, str]:
        with open(labels_path, "r") as f:
            names = yaml.safe_load(f)["names"]
        return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}

    @staticmethod
    def _load_model(model_path: Path) -> cv2.dnn.Net:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def detect(
        self,
        image: np.ndarray,
        input_size: int = 640,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.3,
    ) -> list[dict]:
        """Run detection on a single BGR tile, returning boxes in tile pixel space."""
        height, width = image.shape[:2]
        max_side = max(height, width)
        padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        padded[:height, :width] = image

        blob = cv2.dnn.blobFromImage(padded, 1 / 255, (input_size, input_size), swapRB=True, crop=False)

        with self._lock:
            self.net.setInput(blob)
            preds = self.net.forward()

        scale = max_side / input_size
        boxes, confidences, class_ids = [], [], []

        for det in preds[0]:
            objectness = float(det[4])
            if objectness <= conf_threshold:
                continue

            class_scores = det[5:]
            class_id = int(class_scores.argmax())
            score = objectness * float(class_scores[class_id])
            if score <= conf_threshold:
                continue

            cx, cy, w, h = det[0:4]
            boxes.append(
                [
                    int((cx - 0.5 * w) * scale),
                    int((cy - 0.5 * h) * scale),
                    int(w * scale),
                    int(h * scale),
                ]
            )
            confidences.append(score)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        results = []
        for i in np.array(indices).flatten():
            box = boxes[i]
            results.append(
                {
                    "class_name": self.labels[class_ids[i]],
                    "confidence": confidences[i],
                    "x": box[0],
                    "y": box[1],
                    "width": box[2],
                    "height": box[3],
                }
            )
        return results


def get_gsd(raster_path: Path) -> tuple[float, float]:
    """Return (gsd_x, gsd_y) - the ground sample distance in CRS units/pixel."""
    with rasterio.open(raster_path) as src:
        return abs(src.transform.a), abs(src.transform.e)


def merge_nearby_detections(gdf: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    """Collapse clusters of points within `distance` of each other into a single centroid.

    Deduplicates detections of the same object made in adjacent, overlapping tiles.
    """
    if len(gdf) == 0:
        return gdf

    geometries = list(gdf.geometry)
    tree = STRtree(geometries)

    seen: set[int] = set()
    merged = []
    for idx, point in enumerate(geometries):
        if idx in seen:
            continue

        neighbor_idx = [int(i) for i in tree.query(point, predicate="dwithin", distance=distance) if i not in seen]
        if not neighbor_idx:
            neighbor_idx = [idx]

        centroid = unary_union([geometries[i] for i in neighbor_idx]).centroid
        row = gdf.iloc[idx]
        merged.append({"class_name": row["class_name"], "confidence": row["confidence"], "geometry": centroid})
        seen.update(neighbor_idx)

    return gpd.GeoDataFrame(merged, crs=gdf.crs)
