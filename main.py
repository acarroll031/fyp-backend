import csv
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, Depends, UploadFile, File
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("student_risk_model_0.1-1.0.joblib")
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model():
    return app.state.model

class PredictRequest(BaseModel):
    average_score: float
    assessments_completed: int
    performance_trend: float
    max_consecutive_misses: int
    progress_in_semester: int


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict_risk(
        request: PredictRequest,
        model=Depends(get_model)
):
    features = [[
        request.average_score,
        request.assessments_completed,
        request.performance_trend,
        request.max_consecutive_misses,
        request.progress_in_semester
    ]]
    risk_score = model.predict(features)
    return {"risk_score": risk_score[0]}

@app.get("/students")
def get_students():
    connection = sqlite3.connect("fyp_database.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("SELECT student_id, student_name, module, risk_score FROM risk_scores")
    rows = cursor.fetchall()
    connection.close()

    students = [dict(row) for row in rows]
    return students

@app.post("/students/{module_id}/grades")
async def post_grades(module_id: str, progress_in_semester: int, file: UploadFile = File(...)):
    contents = await file.read()
    csv_data = contents.decode("utf-8")
    csv_reader = csv.DictReader(io.StringIO(csv_data))

    connection = sqlite3.connect("fyp_database.db")
    cursor = connection.cursor()

    for grade in csv_reader:
        cursor.execute(
            "INSERT INTO grades (student_id, student_name, module, assessment_number, score, progress_in_semester) VALUES (?, ?, ?, ?, ?, ?)",
            (
                grade["student_id"],
                grade["student_name"],
                module_id,
                grade["assessment_number"],
                grade["score"],
                progress_in_semester),
        )

    connection.commit()
    connection.close()
    return {"message": "Grades inserted successfully"}

