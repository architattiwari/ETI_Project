from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from app.features import extract_features
from app.models import TRAIN_HINT, ModelsNotTrainedError, load_artifacts, predict
from app.preprocessing import require_text

NO_REFERENCE_WARNING = "No reference translation: WER, TER and length difference were skipped, so no prediction was made."


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.artifacts = load_artifacts()
    except ModelsNotTrainedError:
        app.state.artifacts = None
    yield


app = FastAPI(title="ETI - Evaluating Translation Inaccuracies", lifespan=lifespan)


class PredictRequest(BaseModel):
    source_text: str
    translated_text: str
    reference_text: str | None = None

    @field_validator("source_text", "translated_text")
    @classmethod
    def check_required_text(cls, value: str) -> str:
        return require_text(value)

    @field_validator("reference_text")
    @classmethod
    def check_reference(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return require_text(value)


class PredictResponse(BaseModel):
    features: dict[str, float]
    predictions: dict[str, float] | None = None
    interpretation: str | None = None
    warnings: list[str] = []


def interpret(predictions: dict[str, float]) -> str:
    score = sum(predictions.values()) / len(predictions)
    level = "low" if score < 0.4 else "moderate" if score < 0.7 else "high"
    return (
        f"Average predicted quality is {score:.2f} on a 0-1 scale ({level}). "
        "Lexical features cannot detect meaning errors such as negation or wrong numbers."
    )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": app.state.artifacts is not None}


@app.post("/predict", response_model=PredictResponse)
def predict_quality(request: PredictRequest):
    features = extract_features(request.source_text, request.translated_text, request.reference_text)
    if request.reference_text is None:
        warnings = [NO_REFERENCE_WARNING]
        if app.state.artifacts is None:
            warnings.append(TRAIN_HINT)
        return PredictResponse(features=features, warnings=warnings)

    if app.state.artifacts is None:
        raise HTTPException(status_code=503, detail=f"Models are not available. {TRAIN_HINT}")
    scaler, models = app.state.artifacts
    predictions = {name: float(values[0]) for name, values in predict([features], scaler, models).items()}
    return PredictResponse(features=features, predictions=predictions, interpretation=interpret(predictions))
