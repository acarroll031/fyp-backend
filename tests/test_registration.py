import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from psycopg2 import IntegrityError

from main import app, get_db_connection

# --- Fixtures ---


@pytest.fixture
def mock_connection():
    """Create a mock psycopg2 connection and cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def client(mock_connection):
    """Create a TestClient with the DB connection dependency overridden."""
    app.dependency_overrides[get_db_connection] = lambda: mock_connection
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- Registration Tests ---


class TestRegisterLecturer:
    """Tests for POST /register"""

    def test_register_success(self, client, mock_connection):
        """Test successful lecturer registration."""
        payload = {
            "email": "lecturer@test.com",
            "password": "securepassword",
            "lecturer_name": "Dr. Test",
        }

        response = client.post("/register", json=payload)

        assert response.status_code == 200
        assert response.json() == {"message": "Lecturer registered successfully"}
        mock_connection.cursor().execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    def test_register_duplicate_email_returns_400(self, client, mock_connection):
        """Test that registering with a duplicate email returns 400."""
        mock_connection.cursor().execute.side_effect = IntegrityError

        payload = {
            "email": "duplicate@test.com",
            "password": "securepassword",
            "lecturer_name": "Dr. Duplicate",
        }

        response = client.post("/register", json=payload)

        assert response.status_code == 400
        assert response.json()["detail"] == "Email already registered"
        mock_connection.rollback.assert_called_once()

    def test_register_missing_field_returns_422(self, client):
        """Test that a missing required field returns 422."""
        payload = {
            "email": "lecturer@test.com",
            # missing password and lecturer_name
        }

        response = client.post("/register", json=payload)

        assert response.status_code == 422

    def test_register_empty_body_returns_422(self, client):
        """Test that an empty body returns 422."""
        response = client.post("/register", json={})

        assert response.status_code == 422

    def test_register_connection_closed_on_success(self, client, mock_connection):
        """Test that the DB connection is closed after successful registration."""
        payload = {
            "email": "lecturer@test.com",
            "password": "securepassword",
            "lecturer_name": "Dr. Test",
        }

        client.post("/register", json=payload)

        mock_connection.close.assert_called_once()

    def test_register_connection_closed_on_error(self, client, mock_connection):
        """Test that the DB connection is closed even when an error occurs."""
        mock_connection.cursor().execute.side_effect = IntegrityError

        payload = {
            "email": "duplicate@test.com",
            "password": "password",
            "lecturer_name": "Dr. Duplicate",
        }

        client.post("/register", json=payload)

        mock_connection.close.assert_called_once()


# --- Login Tests ---


class TestLogin:
    """Tests for POST /login"""

    @patch("main.verify_password", return_value=True)
    def test_login_success(self, mock_verify, client, mock_connection):
        """Test successful login returns a JWT token."""
        mock_connection.cursor().fetchone.return_value = {
            "email": "lecturer@test.com",
            "password_hash": "hashed_password",
            "lecturer_name": "Dr. Test",
        }

        response = client.post(
            "/login",
            data={"username": "lecturer@test.com", "password": "securepassword"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @patch("main.verify_password", return_value=False)
    def test_login_wrong_password_returns_401(
        self, mock_verify, client, mock_connection
    ):
        """Test that an incorrect password returns 401."""
        mock_connection.cursor().fetchone.return_value = {
            "email": "lecturer@test.com",
            "password_hash": "hashed_password",
            "lecturer_name": "Dr. Test",
        }

        response = client.post(
            "/login",
            data={"username": "lecturer@test.com", "password": "wrongpassword"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect email or password"

    def test_login_nonexistent_user_returns_401(self, client, mock_connection):
        """Test that a non-existent user returns 401."""
        mock_connection.cursor().fetchone.return_value = None

        response = client.post(
            "/login",
            data={"username": "nobody@test.com", "password": "password"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect email or password"

    def test_login_missing_fields_returns_422(self, client):
        """Test that missing login fields returns 422."""
        response = client.post("/login", data={})

        assert response.status_code == 422
