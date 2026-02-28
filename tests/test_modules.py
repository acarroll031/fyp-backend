import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from psycopg2 import IntegrityError

from main import app, get_db_connection, get_current_user

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
    """Create a TestClient with DB connection and auth dependencies overridden."""
    app.dependency_overrides[get_db_connection] = lambda: mock_connection
    app.dependency_overrides[get_current_user] = lambda: "lecturer@test.com"
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- POST /modules Tests ---


class TestCreateModule:
    """Tests for POST /modules"""

    def test_create_module_success(self, client, mock_connection):
        """Test successful module creation."""
        payload = {
            "module_name": "Intro to Programming",
            "module_code": "CS161",
            "assessment_count": 10,
        }

        response = client.post("/modules", json=payload)

        assert response.status_code == 200
        assert response.json() == {"message": "Module CS161 created successfully"}
        mock_connection.cursor().execute.assert_called_once()
        mock_connection.commit.assert_called_once()

    def test_create_module_duplicate_code_returns_400(self, client, mock_connection):
        """Test that creating a module with an existing code returns 400."""
        mock_connection.cursor().execute.side_effect = IntegrityError

        payload = {
            "module_name": "Intro to Programming",
            "module_code": "CS161",
            "assessment_count": 10,
        }

        response = client.post("/modules", json=payload)

        assert response.status_code == 400
        assert response.json()["detail"] == "Module code already exists"
        mock_connection.rollback.assert_called_once()

    def test_create_module_missing_field_returns_422(self, client):
        """Test that a missing required field returns 422."""
        payload = {
            "module_name": "Intro to Programming",
            # missing module_code and assessment_count
        }

        response = client.post("/modules", json=payload)

        assert response.status_code == 422

    def test_create_module_empty_body_returns_422(self, client):
        """Test that an empty body returns 422."""
        response = client.post("/modules", json={})

        assert response.status_code == 422

    def test_create_module_connection_closed_on_success(self, client, mock_connection):
        """Test that the DB connection is closed after successful creation."""
        payload = {
            "module_name": "Intro to Programming",
            "module_code": "CS161",
            "assessment_count": 10,
        }

        client.post("/modules", json=payload)

        mock_connection.close.assert_called_once()

    def test_create_module_connection_closed_on_error(self, client, mock_connection):
        """Test that the DB connection is closed even when an error occurs."""
        mock_connection.cursor().execute.side_effect = IntegrityError

        payload = {
            "module_name": "Intro to Programming",
            "module_code": "CS161",
            "assessment_count": 10,
        }

        client.post("/modules", json=payload)

        mock_connection.close.assert_called_once()

    def test_create_module_requires_auth(self, mock_connection):
        """Test that the endpoint requires authentication."""
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides.pop(get_current_user, None)
        unauthenticated_client = TestClient(app)

        payload = {
            "module_name": "Intro to Programming",
            "module_code": "CS161",
            "assessment_count": 10,
        }

        response = unauthenticated_client.post("/modules", json=payload)

        assert response.status_code == 401
        app.dependency_overrides.clear()


# --- GET /modules Tests ---


class TestGetModules:
    """Tests for GET /modules"""

    def test_get_modules_returns_list(self, client, mock_connection):
        """Test that the endpoint returns a list of modules."""
        mock_connection.cursor().fetchall.return_value = [
            {
                "module_code": "CS161",
                "module_name": "Intro to Programming",
                "lecturer_email": "lecturer@test.com",
                "assessment_count": 10,
            },
            {
                "module_code": "CS260",
                "module_name": "Data Structures",
                "lecturer_email": "lecturer@test.com",
                "assessment_count": 8,
            },
        ]

        response = client.get("/modules")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["module_code"] == "CS161"
        assert data[1]["module_name"] == "Data Structures"

    def test_get_modules_empty_list(self, client, mock_connection):
        """Test that the endpoint returns an empty list when no modules exist."""
        mock_connection.cursor().fetchall.return_value = []

        response = client.get("/modules")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_modules_connection_closed(self, client, mock_connection):
        """Test that the DB connection is closed after the request."""
        mock_connection.cursor().fetchall.return_value = []

        client.get("/modules")

        mock_connection.close.assert_called_once()

    def test_get_modules_requires_auth(self, mock_connection):
        """Test that the endpoint requires authentication."""
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides.pop(get_current_user, None)
        unauthenticated_client = TestClient(app)

        response = unauthenticated_client.get("/modules")

        assert response.status_code == 401
        app.dependency_overrides.clear()


# --- DELETE /modules/{module_code} Tests ---


class TestDeleteModule:
    """Tests for DELETE /modules/{module_code}"""

    def test_delete_module_success(self, client, mock_connection):
        """Test successful module deletion."""
        # fetchone returns a row indicating the module exists
        mock_connection.cursor().fetchone.return_value = {
            "module_code": "CS161",
            "module_name": "Intro to Programming",
            "lecturer_email": "lecturer@test.com",
            "assessment_count": 10,
        }

        response = client.delete("/modules/CS161")

        assert response.status_code == 200
        assert response.json() == {"message": "Module CS161 deleted successfully"}
        mock_connection.commit.assert_called_once()

    def test_delete_module_not_found_returns_404(self, client, mock_connection):
        """Test that deleting a non-existent module returns 404."""
        mock_connection.cursor().fetchone.return_value = None

        response = client.delete("/modules/NONEXISTENT")

        assert response.status_code == 404
        assert response.json()["detail"] == "Module not found or access denied"

    def test_delete_module_connection_closed_on_success(self, client, mock_connection):
        """Test that the DB connection is closed after successful deletion."""
        mock_connection.cursor().fetchone.return_value = {
            "module_code": "CS161",
        }

        client.delete("/modules/CS161")

        mock_connection.close.assert_called_once()

    def test_delete_module_connection_closed_on_not_found(
        self, client, mock_connection
    ):
        """Test that the DB connection is closed when module is not found."""
        mock_connection.cursor().fetchone.return_value = None

        client.delete("/modules/NONEXISTENT")

        mock_connection.close.assert_called_once()

    def test_delete_module_requires_auth(self, mock_connection):
        """Test that the endpoint requires authentication."""
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides.pop(get_current_user, None)
        unauthenticated_client = TestClient(app)

        response = unauthenticated_client.delete("/modules/CS161")

        assert response.status_code == 401
        app.dependency_overrides.clear()
