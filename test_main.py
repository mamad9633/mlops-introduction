from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_list_models():
    with TestClient(app) as client:
        response = client.get("/models")
        assert response.status_code == 200
        assert response.json() == {"available_models": ["logistic_model", "rf_model"]}


def test_predict_invalid_model():
    with TestClient(app) as client:
        response = client.post(
            "/predict/invalid_model",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )

        assert response.status_code == 422


@pytest.fixture
def mock_models(mocker):
    mock_dict = {"logistic_model": MagicMock, "rf_model": MagicMock}
    m = mocker.patch(
        "app.main.ml_models",
        return_value=mock_dict,
    )
    m.keys.return_value = mock_dict.keys()
    return m


def test_predict_mocked(mock_models):
    mock_models["logistic_model"].predict.return_value = [-1]

    with TestClient(app) as client:
        response = client.post(
            "/predict/logistic_model",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )

        assert response.status_code == 200
        assert response.json() == {"model": "logistic_model", "prediction": -1}