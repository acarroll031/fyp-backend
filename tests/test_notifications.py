import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

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


# --- GET /notifications Tests ---


class TestGetNotifications:
    """Tests for GET /notifications"""

    def test_get_notifications_returns_list(self, client, mock_connection):
        """Test that the endpoint returns a list of notifications."""
        mock_connection.cursor().fetchall.return_value = [
            {
                "id": 1,
                "lecturer_email": "lecturer@test.com",
                "message": '{"text": "Student Alice is at risk"}',
                "is_read": False,
                "created_at": "2026-02-28T10:00:00",
                "notification_type": "RISK_ALERT",
                "module": "CS161",
            },
            {
                "id": 2,
                "lecturer_email": "lecturer@test.com",
                "message": '{"text": "Grades uploaded successfully"}',
                "is_read": True,
                "created_at": "2026-02-27T10:00:00",
                "notification_type": "UPLOAD_SUCCESS",
                "module": "CS161",
            },
        ]

        response = client.get("/notifications")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[0]["notification_type"] == "RISK_ALERT"
        assert data[1]["is_read"] is True

    def test_get_notifications_empty_list(self, client, mock_connection):
        """Test that the endpoint returns an empty list when no notifications exist."""
        mock_connection.cursor().fetchall.return_value = []

        response = client.get("/notifications")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_notifications_connection_closed(self, client, mock_connection):
        """Test that the DB connection is closed after the request."""
        mock_connection.cursor().fetchall.return_value = []

        client.get("/notifications")

        mock_connection.close.assert_called_once()

    def test_get_notifications_requires_auth(self, mock_connection):
        """Test that the endpoint requires authentication."""
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides.pop(get_current_user, None)
        unauthenticated_client = TestClient(app)

        response = unauthenticated_client.get("/notifications")

        assert response.status_code == 401
        app.dependency_overrides.clear()


# --- PUT /notifications/{notification_id}/read Tests ---


class TestMarkNotificationAsRead:
    """Tests for PUT /notifications/{notification_id}/read"""

    def test_mark_as_read_success(self, client, mock_connection):
        """Test successfully marking a notification as read."""
        mock_connection.cursor().fetchone.return_value = {
            "id": 1,
            "lecturer_email": "lecturer@test.com",
            "is_read": False,
        }

        response = client.put("/notifications/1/read")

        assert response.status_code == 200
        assert response.json() == {"message": "Notification marked as read"}
        mock_connection.commit.assert_called_once()

    def test_mark_as_read_not_found_returns_404(self, client, mock_connection):
        """Test that a 404 is returned when the notification is not found."""
        mock_connection.cursor().fetchone.return_value = None

        response = client.put("/notifications/9999/read")

        assert response.status_code == 404
        assert response.json()["detail"] == "Notification not found or access denied"

    def test_mark_as_read_connection_closed(self, client, mock_connection):
        """Test that the DB connection is closed after the request."""
        mock_connection.cursor().fetchone.return_value = {
            "id": 1,
            "lecturer_email": "lecturer@test.com",
            "is_read": False,
        }

        client.put("/notifications/1/read")

        mock_connection.close.assert_called_once()

    def test_mark_as_read_requires_auth(self, mock_connection):
        """Test that the endpoint requires authentication."""
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides.pop(get_current_user, None)
        unauthenticated_client = TestClient(app)

        response = unauthenticated_client.put("/notifications/1/read")

        assert response.status_code == 401
        app.dependency_overrides.clear()


# --- PUT /notifications/{notification_id}/unread Tests ---


class TestMarkNotificationAsUnread:
    """Tests for PUT /notifications/{notification_id}/unread"""

    def test_mark_as_unread_success(self, client, mock_connection):
        """Test successfully marking a notification as unread."""
        mock_connection.cursor().fetchone.return_value = {
            "id": 1,
            "lecturer_email": "lecturer@test.com",
            "is_read": True,
        }

        response = client.put("/notifications/1/unread")

        assert response.status_code == 200
        assert response.json() == {"message": "Notification marked as unread"}
        mock_connection.commit.assert_called_once()

    def test_mark_as_unread_not_found_returns_404(self, client, mock_connection):
        """Test that a 404 is returned when the notification is not found."""
        mock_connection.cursor().fetchone.return_value = None

        response = client.put("/notifications/9999/unread")

        assert response.status_code == 404
        assert response.json()["detail"] == "Notification not found or access denied"

    def test_mark_as_unread_connection_closed(self, client, mock_connection):
        """Test that the DB connection is closed after the request."""
        mock_connection.cursor().fetchone.return_value = {
            "id": 1,
            "lecturer_email": "lecturer@test.com",
            "is_read": True,
        }

        client.put("/notifications/1/unread")

        mock_connection.close.assert_called_once()

    def test_mark_as_unread_requires_auth(self, mock_connection):
        """Test that the endpoint requires authentication."""
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides.pop(get_current_user, None)
        unauthenticated_client = TestClient(app)

        response = unauthenticated_client.put("/notifications/1/unread")

        assert response.status_code == 401
        app.dependency_overrides.clear()
