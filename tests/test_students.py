import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from main import app, get_db_connection, get_current_user, get_model

# --- Fixtures ---


@pytest.fixture
def mock_connection():
    """Create a mock psycopg2 connection and cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def mock_model():
    """Create a mock ML model."""
    model = MagicMock()
    return model


@pytest.fixture
def client(mock_connection, mock_model):
    """Create a TestClient with DB connection, auth, and model dependencies overridden."""
    app.dependency_overrides[get_db_connection] = lambda: mock_connection
    app.dependency_overrides[get_current_user] = lambda: "lecturer@test.com"
    app.dependency_overrides[get_model] = lambda: mock_model
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- GET /students Tests ---


class TestGetStudents:
    """Tests for GET /students"""

    def test_get_students_returns_list(self, client, mock_connection):
        """Test that the endpoint returns a list of students."""
        mock_connection.cursor().fetchall.return_value = [
            {
                "student_id": 1000,
                "student_name": "Alice Murphy",
                "module": "CS161",
                "risk_score": 72.5,
                "previous_risk_score": 60.0,
            },
            {
                "student_id": 1001,
                "student_name": "Bob Kelly",
                "module": "CS161",
                "risk_score": 35.0,
                "previous_risk_score": 40.0,
            },
        ]

        response = client.get("/students")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["student_id"] == 1000
        assert data[1]["student_name"] == "Bob Kelly"

    def test_get_students_empty_list(self, client, mock_connection):
        """Test that the endpoint returns an empty list when no students exist."""
        mock_connection.cursor().fetchall.return_value = []

        response = client.get("/students")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_students_connection_closed(self, client, mock_connection):
        """Test that the DB connection is closed after the request."""
        mock_connection.cursor().fetchall.return_value = []

        client.get("/students")

        mock_connection.close.assert_called_once()

    def test_get_students_requires_auth(self, mock_connection):
        """Test that the endpoint requires authentication."""
        # Remove the auth override so the real dependency runs
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides.pop(get_current_user, None)
        unauthenticated_client = TestClient(app)

        response = unauthenticated_client.get("/students")

        assert response.status_code == 401
        app.dependency_overrides.clear()


# --- GET /students/{student_id}/{module_id} Tests ---


class TestGetStudentDetailsByModule:
    """Tests for GET /students/{student_id}/{module_id}"""

    def test_get_student_details_success(self, client, mock_connection):
        """Test that valid student details are returned."""
        cursor = mock_connection.cursor()
        cursor.fetchone.return_value = {
            "student_id": 1000,
            "student_name": "Alice Murphy",
            "module": "CS161",
            "average_score": 65.5,
            "assessments_completed": 5,
            "performance_trend": 2.1,
            "max_consecutive_misses": 0,
            "progress_in_semester": 0.5,
            "risk_score": 72.5,
        }
        # fetchall is called twice: once for grades, once for risk_history
        cursor.fetchall.side_effect = [
            # grades
            [
                {"assessment_number": 1, "score": 70.0, "progress_in_semester": 0.1},
                {"assessment_number": 2, "score": 61.0, "progress_in_semester": 0.2},
            ],
            # risk_history
            [
                {
                    "risk_score_history": 60.0,
                    "risk_score_history_timestamp": "2026-01-15T10:00:00",
                },
                {
                    "risk_score_history": 72.5,
                    "risk_score_history_timestamp": "2026-02-15T10:00:00",
                },
            ],
        ]

        response = client.get("/students/1000/CS161")

        assert response.status_code == 200
        data = response.json()
        assert data["student"]["student_id"] == 1000
        assert len(data["grades"]) == 2
        assert len(data["risk_history"]) == 2

    def test_get_student_details_not_found(self, client, mock_connection):
        """Test that a 404 is returned when the student is not found."""
        mock_connection.cursor().fetchone.return_value = None

        response = client.get("/students/9999/CS161")

        assert response.status_code == 404
        assert response.json()["detail"] == "Student not found or access denied"

    def test_get_student_details_connection_closed(self, client, mock_connection):
        """Test that the DB connection is closed after the request."""
        mock_connection.cursor().fetchone.return_value = None

        client.get("/students/9999/CS161")

        mock_connection.close.assert_called_once()


# --- POST /students/{module_id}/grades Tests ---


class TestPostGrades:
    """Tests for POST /students/{module_id}/grades"""

    @patch("main.pd.read_sql_query")
    @patch("main.convert_grades_to_students")
    def test_post_grades_success(
        self, mock_convert, mock_read_sql, client, mock_connection, mock_model
    ):
        """Test successful grade upload and risk score calculation."""
        import pandas as pd
        import numpy as np

        # Mock the CSV content
        csv_content = (
            "student_id,student_name,email,assessment_number,score\n"
            "1000,Alice Murphy,alice@test.com,1,70.0\n"
            "1001,Bob Kelly,bob@test.com,1,55.0\n"
        )

        # Mock convert_grades_to_students return
        students_df = pd.DataFrame(
            {
                "student_id": [1000, 1001],
                "student_name": ["Alice Murphy", "Bob Kelly"],
                "module": ["CS161", "CS161"],
                "average_score": [70.0, 55.0],
                "assessments_completed": [1, 1],
                "performance_trend": [0.0, 0.0],
                "max_consecutive_misses": [0, 0],
                "progress_in_semester": [0.1, 0.1],
            }
        )
        mock_convert.return_value = students_df

        # Mock pd.read_sql_query for existing grades and risk scores
        mock_read_sql.side_effect = [
            # grades query
            pd.DataFrame(
                {
                    "student_id": [1000, 1001],
                    "student_name": ["Alice Murphy", "Bob Kelly"],
                    "module": ["CS161", "CS161"],
                    "assessment_number": [1, 1],
                    "score": [70.0, 55.0],
                    "progress_in_semester": [0.1, 0.1],
                }
            ),
            # existing risk_scores query
            pd.DataFrame(
                {
                    "student_id": pd.Series(dtype="int"),
                    "risk_score": pd.Series(dtype="float"),
                }
            ),
        ]

        # Mock model predictions
        mock_model.predict.return_value = np.array([30.0, 65.0])

        # Mock fetchone for lecturer_email
        mock_connection.cursor().fetchone.return_value = ("lecturer@test.com",)

        response = client.post(
            "/students/CS161/grades?progress_in_semester=0.1",
            files={"file": ("grades.csv", csv_content, "text/csv")},
        )

        assert response.status_code == 200
        assert (
            response.json()["message"]
            == "Grades inserted and risk scores updated successfully"
        )
        mock_model.predict.assert_called_once()

    def test_post_grades_no_file_returns_422(self, client):
        """Test that missing file returns 422."""
        response = client.post("/students/CS161/grades?progress_in_semester=0.1")

        assert response.status_code == 422

    @patch("main.pd.read_sql_query")
    @patch("main.convert_grades_to_students")
    def test_post_grades_rollback_on_error(
        self, mock_convert, mock_read_sql, client, mock_connection, mock_model
    ):
        """Test that the DB is rolled back when an exception occurs."""
        mock_convert.side_effect = Exception("Processing error")

        csv_content = (
            "student_id,student_name,email,assessment_number,score\n"
            "1000,Alice Murphy,alice@test.com,1,70.0\n"
        )

        # Mock pd.read_sql_query to return a valid DataFrame before the error
        import pandas as pd

        mock_read_sql.return_value = pd.DataFrame(
            {
                "student_id": [1000],
                "student_name": ["Alice Murphy"],
                "module": ["CS161"],
                "assessment_number": [1],
                "score": [70.0],
                "progress_in_semester": [0.1],
            }
        )

        # Use a client that doesn't raise server exceptions so we can inspect the 500 response
        app.dependency_overrides[get_db_connection] = lambda: mock_connection
        app.dependency_overrides[get_current_user] = lambda: "lecturer@test.com"
        app.dependency_overrides[get_model] = lambda: mock_model
        error_client = TestClient(app, raise_server_exceptions=False)

        response = error_client.post(
            "/students/CS161/grades?progress_in_semester=0.1",
            files={"file": ("grades.csv", csv_content, "text/csv")},
        )

        assert response.status_code == 500
        mock_connection.rollback.assert_called_once()
