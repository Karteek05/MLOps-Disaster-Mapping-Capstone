from pydantic import BaseModel


class ClassBreakdown(BaseModel):
    id: int
    name: str
    color: str  # "#RRGGBB"
    pixel_count: int
    percentage: float


class PredictResponse(BaseModel):
    overlay_image: str  # data:image/png;base64,...
    img_size: int
    classes: list[ClassBreakdown]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
