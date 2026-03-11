# Predictive Analytics for Student Success — Backend

> **FYP25AM006** — Final Year Project  
> A FastAPI backend that uses machine learning to predict student risk scores based on assessment performance, helping lecturers identify and support at-risk students early.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Machine Learning Model](#machine-learning-model)
  - [Features Used](#features-used)
  - [Training](#training)
- [Seeding & Test Data](#seeding--test-data)
- [Running Tests](#running-tests)

---

## Overview

This backend powers a web application that allows lecturers to:

1. **Upload student grades** (CSV) for their modules.
2. **Automatically calculate risk scores** for each student using a trained ML model.
3. **View student details**, including grade history, risk score trends, and performance metrics.
4. **Receive notifications** when a student becomes newly at risk.

Risk scores are predicted from features derived from assessment data — such as average score, performance trend, consecutive missed assessments, and semester progress.

## Features

- **JWT Authentication** — Lecturer registration and login with hashed passwords (bcrypt) and JWT tokens.
- **Module Management** — Create, read, update, and delete modules.
- **Grade Upload** — Upload CSV files of student grades per module; grades are upserted into the database.
- **Risk Prediction** — An XGBoost regression model predicts a risk score (0–100) for each student after every grade upload.
- **Risk History Tracking** — Risk scores are recorded over time so lecturers can view trends.
- **Notifications** — Automatic alerts when a student's risk score crosses the "at risk" threshold (>70) for the first time.
- **REST API** — All functionality exposed via a RESTful API with interactive Swagger docs.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Language | Python 3.13 |
| Database | PostgreSQL (via [psycopg2](https://www.psycopg.org/)) |
| ML Models | [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/), Random Forest |
| Auth | JWT ([PyJWT](https://pyjwt.readthedocs.io/)), bcrypt |
| Data Processing | [pandas](https://pandas.pydata.org/), NumPy |
| Server | [Uvicorn](https://www.uvicorn.org/) |

## Project Structure

```
fyp-backend/
├── main.py                    # FastAPI application — all API endpoints
├── data_processing.py         # Data preprocessing, feature engineering, and training data generation
├── requirements.txt           # Python dependencies
├── schema_definition_dump.sql # PostgreSQL schema definition
│
├── model_training/
│   └── model_training.py      # Model training scripts (Random Forest, XGBoost, KNN)
│
├── student_risk_model_0.1-1.0.joblib   # Trained XGBoost model (loaded at startup)
├── student_risk_model_RF_0.1-1.0.joblib # Trained Random Forest model
│
├── training_data/             # CSV training datasets at various progress thresholds
├── sample_assessment_data/    # Sample CSV files for seeding grade uploads
│
├── generate_test_data.py      # Generate synthetic assessment CSVs with configurable miss rates
├── seed_assessments.py        # Automate uploading sample assessment CSVs to the API
├── seed_synthetic_tags.py     # Seed the database with synthetic students for each risk tag
│
└── tests/                     # Pytest test suite
    ├── test_predict.py        # Tests for /predict endpoint
    ├── test_students.py       # Tests for student endpoints
    ├── test_modules.py        # Tests for module CRUD endpoints
    ├── test_notifications.py  # Tests for notification endpoints
    └── test_registration.py   # Tests for registration/login endpoints
```

## Database Schema

The PostgreSQL database contains the following tables (see `schema_definition_dump.sql` for full DDL):

| Table | Purpose |
|---|---|
| `lecturers` | Lecturer accounts (email, name, hashed password) |
| `modules` | Modules with code, name, assessment count, and linked lecturer |
| `students` | Per-module student records with computed features (avg score, trend, etc.) |
| `grades` | Individual assessment grades per student per module |
| `risk_scores` | Current and previous risk scores per student per module |
| `risk_history` | Time-series log of all risk score changes |
| `notifications` | Alerts for lecturers (risk alerts, upload confirmations) |

## Getting Started

### Prerequisites

- **Python 3.13+**
- **PostgreSQL** database (local or cloud-hosted, e.g. [Neon](https://neon.tech/))
- **pip** (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://gitlab.cs.nuim.ie/u230473/fyp25am006-predictive-analytics-for-student-success.git
   cd fyp25am006-predictive-analytics-for-student-success
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database:**

   Create a PostgreSQL database and run the schema definition:

   ```bash
   psql -d <your_database> -f schema_definition_dump.sql
   ```

### Environment Variables

Create a `.env` file in the project root with the following:

```env
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<database>
```

### Running the Server

Start the FastAPI development server with Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at **http://127.0.0.1:8000**.

- **Swagger UI (interactive docs):** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

## API Endpoints

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/register` | Register a new lecturer | No |
| `POST` | `/login` | Login and receive a JWT token | No |
| `POST` | `/predict` | Predict a student's risk score from features | Yes |
| `GET` | `/students` | List students for the logged-in lecturer | Yes |
| `GET` | `/students/{student_id}/{module_id}` | Get detailed student info (grades, risk history) | Yes |
| `POST` | `/students/{module_id}/grades` | Upload grades CSV and update risk scores | No |
| `POST` | `/modules` | Create a new module | Yes |
| `GET` | `/modules` | List modules for the logged-in lecturer | Yes |
| `PUT` | `/modules/{module_code}` | Update a module | Yes |
| `DELETE` | `/modules/{module_code}` | Delete a module | Yes |
| `GET` | `/notifications` | Get notifications for the logged-in lecturer | Yes |
| `PUT` | `/notifications/{id}/read` | Mark a notification as read | Yes |
| `PUT` | `/notifications/{id}/unread` | Mark a notification as unread | Yes |

### Grade CSV Format

The CSV file uploaded to `/students/{module_id}/grades` should have the following columns:

| Column | Type | Description |
|---|---|---|
| `student_id` | int | Unique student identifier |
| `student_name` | str | Student's full name |
| `email` | str | Student's email address |
| `assessment_number` | int | Assessment number (e.g. 1, 2, 3…) |
| `score` | float | Score achieved (0–100) |

## Machine Learning Model

The application loads a pre-trained **XGBoost Regressor** model (`student_risk_model_0.1-1.0.joblib`) at startup. This model predicts a **risk score (0–100)** for each student, where higher scores indicate greater risk of academic failure.

### Features Used

| Feature | Description |
|---|---|
| `average_score` | Mean score across all completed assessments |
| `assessments_completed` | Number of assessments with a non-zero score |
| `performance_trend` | Difference between average of second-half and first-half scores |
| `max_consecutive_misses` | Longest streak of consecutive zero-score assessments |
| `progress_in_semester` | Proportion of semester completed (0.0–1.0) |

### Training

Model training scripts are located in `model_training/model_training.py`. Three model types are supported:

- **Random Forest Regressor** — baseline model
- **XGBoost Regressor** — primary model (best performance)
- **K-Nearest Neighbours Regressor** — alternative model

Training uses **GroupShuffleSplit** (grouped by Student ID) to prevent data leakage, and **GridSearchCV** for hyperparameter tuning. To retrain:

```bash
cd model_training
python model_training.py
```

Training data is generated from `data_processing.py` at various semester progress thresholds (0.1 to 1.0) and combined into a single dataset (`training_data/Student_Data_training_0.1-1.0.csv`).

## Seeding & Test Data

Several utility scripts are provided for populating the database with sample data:

- **`seed_assessments.py`** — Automates uploading sample assessment CSVs (from `sample_assessment_data/`) to the running API. Configure `MODULE_CODE`, `CSV_FOLDER`, and `TOTAL_ASSESSMENTS` at the top of the file, then run:

  ```bash
  python seed_assessments.py
  ```

- **`seed_synthetic_tags.py`** — Seeds the database directly with synthetic students that cover each risk tag category (Newly At Risk, At Risk, Improving, On Track).

  ```bash
  python seed_synthetic_tags.py
  ```

- **`generate_test_data.py`** — Generates synthetic assessment CSV files with configurable missing rates for testing purposes.

  ```bash
  python generate_test_data.py
  ```

## Running Tests

The test suite uses **pytest** with mocked database connections and ML models.

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_predict.py
```

To run with verbose output:

```bash
pytest -v
```
