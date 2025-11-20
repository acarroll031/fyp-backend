import csv
import io
from contextlib import asynccontextmanager
from dataProcessing import convert_grades_to_students

import pandas as pd
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
    progress_in_semester: float


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
async def post_grades(
        module_id: str,
        progress_in_semester: float,
        file: UploadFile = File(...),
        model=Depends(get_model)
):
    connection = sqlite3.connect("fyp_database.db", timeout=10.0)
    cursor = connection.cursor()

    try:
        contents = await file.read()
        grades_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        grades_df["module"] = module_id
        grades_df["progress_in_semester"] = progress_in_semester
        print("grades_df: ",grades_df.head())

        grade_cols = [
            'student_id', 'student_name', 'assessment_number',
            'score', 'module', 'progress_in_semester'
        ]
        grades_data = grades_df[grade_cols].to_records(index=False).tolist()
        cursor.executemany("""
                    INSERT OR REPLACE INTO grades (
                        student_id, student_name, assessment_number, 
                        score, module, progress_in_semester
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, grades_data)
        connection.commit()

        total_grades_df = pd.read_sql_query("SELECT * FROM grades", connection)
        print("total_grades_df: ",total_grades_df.head())

        student_df = convert_grades_to_students(total_grades_df)
        print("student_df: ", student_df.head())

        student_data = student_df.to_records(index=False).tolist()
        cursor.executemany('''
            INSERT OR REPLACE INTO students (
                student_id, student_name, module, average_score, assessments_completed, performance_trend, max_consecutive_misses, progress_in_semester
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', student_data)
        connection.commit()

        features = student_df[[
            "average_score",
            "assessments_completed",
            "performance_trend",
            "max_consecutive_misses",
            "progress_in_semester"
        ]].values
        risk_scores = model.predict(features)
        student_df["risk_score"] = risk_scores
        print("student_df: ",student_df.head())
        risk_scores_df = student_df[[
            "student_id",
            "student_name",
            "module",
            "risk_score"
        ]]
        risk_scores_df['risk_score'] = risk_scores_df['risk_score'].round(2)
        print("risk_scores_df: ",risk_scores_df.head())
        risk_data = risk_scores_df.to_records(index=False).tolist()
        cursor.executemany('''
            INSERT OR REPLACE INTO risk_scores (student_id, student_name, module, risk_score)
            VALUES (?, ?, ?, ?)
        ''', risk_data)
        connection.commit()

        return {"message": "Grades inserted successfully"}

    except Exception as e:
        connection.rollback()
        raise e
    finally:
        connection.close()



