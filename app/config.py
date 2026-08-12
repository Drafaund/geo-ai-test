"""Runtime configuration, overridable via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PALM_API_")

    model_path: Path = BASE_DIR / "palmCounting-model.onnx"
    labels_path: Path = BASE_DIR / "data.yaml"

    tile_width: int = 640
    tile_height: int = 640

    conf_threshold: float = 0.3
    nms_threshold: float = 0.3
    min_distance: float = 3.0


settings = Settings()
