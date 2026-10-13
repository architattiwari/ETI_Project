import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from app.features import FEATURE_NAMES
from app.main import app
from app.models import ModelsNotTrainedError, load_artifacts

PAYLOAD = {"source_text": "The cat sleeps.", "translated_text": "El gato duerme.", "reference_text": "El gato duerme."}


class ConstantModel:
    def predict(self, x, verbose=0):
        return np.full((len(x), 1), 0.5)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ETI_ARTIFACTS_DIR", str(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_missing_models(client):
    assert client.get("/health").json() == {"status": "ok", "models_loaded": False}


def test_index_page_is_served(client):
    assert client.get("/").status_code == 200


def test_predict_without_models_explains_how_to_train(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 503
    assert "scripts.train" in response.json()["detail"]


def test_predict_without_reference_returns_features_and_warning(client):
    response = client.post("/predict", json={**PAYLOAD, "reference_text": " "})
    body = response.json()
    assert response.status_code == 200
    assert body["predictions"] is None
    assert "wer" not in body["features"]
    assert body["warnings"]


@pytest.mark.parametrize("field", ["source_text", "translated_text"])
def test_predict_rejects_blank_text(client, field):
    assert client.post("/predict", json={**PAYLOAD, field: "  "}).status_code == 422


def test_predict_returns_both_model_scores(client):
    client.app.state.artifacts = (
        StandardScaler().fit(np.zeros((2, len(FEATURE_NAMES)))),
        {"ETI-Baseline": ConstantModel(), "ETI-5": ConstantModel()},
    )
    body = client.post("/predict", json=PAYLOAD).json()
    assert body["predictions"] == {"ETI-Baseline": 0.5, "ETI-5": 0.5}
    assert body["features"]["wer"] == 0


def test_load_artifacts_fails_clearly_when_untrained(tmp_path):
    with pytest.raises(ModelsNotTrainedError, match="scripts.train"):
        load_artifacts(tmp_path)
