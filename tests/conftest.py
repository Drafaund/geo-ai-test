import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402

SAMPLE_RASTER = PROJECT_ROOT / "sawit.tif"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
