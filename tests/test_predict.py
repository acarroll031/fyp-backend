import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from main import app, get_model

# --- Fixtures ---


@pytest.fixture
def mock_model():
    """Create a mock ML model that returns a predictable risk score."""
    model = MagicMock()
    model.predict.return_value = np.array([72.5])
    return model


@pytest.fixture
def client(mock_model):
    """Create a TestClient with the ML model dependency overridden."""
    app.dependency_overrides[get_model] = lambda: mock_model
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- Tests ---


class TestPredictRisk:
    """Tests for POST /predict"""

    def test_predict_returns_risk_score(self, client, mock_model):
        """Test that a valid request returns a predicted risk score."""
        payload = {
            "average_score": 55.0,
            "assessments_completed": 5,
            "performance_trend": -3.2,
            "max_consecutive_misses": 2,
            "progress_in_semester": 0.5,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json() == {"risk_score": 72.5}
        mock_model.predict.assert_called_once()

    def test_predict_passes_correct_features_to_model(self, client, mock_model):
        """Test that the features are passed to the model in the correct order."""
        payload = {
            "average_score": 80.0,
            "assessments_completed": 10,
            "performance_trend": 5.0,
            "max_consecutive_misses": 0,
            "progress_in_semester": 1.0,
        }

        client.post("/predict", json=payload)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [[80.0, 10, 5.0, 0, 1.0]]

    def test_predict_missing_field_returns_422(self, client):
        """Test that a missing required field returns a 422 validation error."""
        payload = {
            "average_score": 55.0,
            # missing assessments_completed
            "performance_trend": -3.2,
            "max_consecutive_misses": 2,
            "progress_in_semester": 0.5,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422

    def test_predict_invalid_type_returns_422(self, client):
        """Test that an invalid field type returns a 422 validation error."""
        payload = {
            "average_score": "not_a_number",
            "assessments_completed": 5,
            "performance_trend": -3.2,
            "max_consecutive_misses": 2,
            "progress_in_semester": 0.5,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, client):
        """Test that an empty request body returns a 422 validation error."""
        response = client.post("/predict", json={})

        assert response.status_code == 422
