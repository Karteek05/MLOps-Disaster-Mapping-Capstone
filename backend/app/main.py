import base64
import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from .inference import CLASS_NAMES, COLOR_MAP, get_model, predict_damage
from .schemas import ClassBreakdown, HealthResponse, PredictResponse

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
    except Exception:
        logger.exception("Model failed to load at startup")
    yield


app = FastAPI(title="Disaster Damage Mapping API", lifespan=lifespan)

CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        model_loaded = get_model() is not None
    except Exception:
        model_loaded = False
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(
    pre_disaster_image: UploadFile = File(...),
    post_disaster_image: UploadFile = File(...),
):
    try:
        pre_bytes = pre_disaster_image.file.read()
        post_bytes = post_disaster_image.file.read()
        blended, counts, img_size = predict_damage(pre_bytes, post_bytes)
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode one of the uploaded images.")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(503, "Model not available or prediction failed.")

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    total_pixels = img_size * img_size
    classes = [
        ClassBreakdown(
            id=class_id,
            name=CLASS_NAMES[class_id],
            color=_hex_color(COLOR_MAP[class_id]),
            pixel_count=counts[class_id],
            percentage=(counts[class_id] / total_pixels) * 100,
        )
        for class_id in sorted(CLASS_NAMES)
    ]

    return PredictResponse(
        overlay_image=f"data:image/png;base64,{overlay_b64}",
        img_size=img_size,
        classes=classes,
    )
