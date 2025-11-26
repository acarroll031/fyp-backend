import io
from contextlib import asynccontextmanager
from dataProcessing import convert_grades_to_students

import pandas as pd
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

from datetime import datetime, timedelta
from typing import Optional
import bcrypt
import jwt # For the tokens
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

SECRET_KEY = "secret"
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_password_hash(password: str) -> str:

    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_byte_encoded = plain_password.encode('utf-8')
    hashed_byte_encoded = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte_encoded, hashed_byte_encoded)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    return email

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("student_risk_model_0.1-1.0.joblib")
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)

origins = ["*"]

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
def get_students(current_user_email: str = Depends(get_current_user)):
    connection = sqlite3.connect("fyp_database.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    query="""
        SELECT s.student_id, s.student_name, s.module, s.risk_score
        FROM risk_scores s
        JOIN modules m ON s.module = m.module_code
        WHERE m.lecturer_email = ?
    """

    cursor.execute(query , (current_user_email,))
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

class LecturerCreate(BaseModel):
    email: str
    password: str
    lecturer_name: str

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/register")
def register_lecturer(lecturer: LecturerCreate):
    connection = sqlite3.connect("fyp_database.db")
    cursor = connection.cursor()

    hashed_password = get_password_hash(lecturer.password)

    try:
        cursor.execute("""
            INSERT INTO lecturers (email, password_hash, lecturer_name)
            VALUES (?, ?, ?)
        """, (lecturer.email, hashed_password, lecturer.lecturer_name)
        )
        connection.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        connection.close()

    return {"message": "Lecturer registered successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    connection = sqlite3.connect("fyp_database.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute(""" SELECT * FROM lecturers WHERE email = ?""", (form_data.username, ))
    lecturer = cursor.fetchone()
    connection.close()

    if not lecturer or not verify_password(form_data.password, lecturer["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": lecturer["email"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}

class ModuleCreate(BaseModel):
    module_name: str
    module_code: str
    assessment_count: int

class ModuleUpdate(BaseModel):
    module_name: Optional[str] = None
    assessment_count: Optional[int] = None

@app.post("/modules")
def create_module(
        module: ModuleCreate,
        current_user: str = Depends(get_current_user)
):
    connection = sqlite3.connect("fyp_database.db")
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO modules (module_name, module_code, assessment_count, lecturer_email) VALUES (?, ?, ?, ?)
        """,
         (module.module_name, module.module_code, module.assessment_count, current_user))
        connection.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Module code already exists")
    finally:
        connection.close()

    return {"message": f"Module {module.module_code} created successfully"}

@app.get("/modules")
def get_modules(
        current_user_email: str = Depends(get_current_user)
):
    connection = sqlite3.connect("fyp_database.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE lecturer_email = ?
    """, (current_user_email,))
    modules = cursor.fetchall()
    connection.close()

    return [dict(row) for row in modules]

@app.put("/modules/{module_code}")
def update_module(
        module_code: str,
        module_update: ModuleUpdate,
        current_user_email: str = Depends(get_current_user)
):
    connection = sqlite3.connect("fyp_database.db")
    cursor = connection.cursor()

    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE module_code = ? AND lecturer_email = ?
    """, (module_code, current_user_email))
    if not cursor.fetchone():
        connection.close()
        raise HTTPException(status_code=404, detail="Module not found or access denied")

    if module_update.module_name:
        cursor.execute(""" UPDATE modules SET module_name = ? WHERE module_code = ? """, (module_update.module_name, module_code))
    if module_update.assessment_count:
        cursor.execute(""" UPDATE modules SET assessment_count = ? WHERE module_code = ? """, (module_update.assessment_count, module_code))

    connection.commit()
    connection.close()
    return {"message": f"Module {module_update.module_code} updated successfully"}

@app.delete("/modules/{module_code}")
def delete_module(
        module_code: str,
        current_user_email: str = Depends(get_current_user)
):
    connection = sqlite3.connect("fyp_database.db")
    cursor = connection.cursor()

    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE module_code = ? AND lecturer_email = ?
    """, (module_code, current_user_email))
    if not cursor.fetchone():
        connection.close()
        raise HTTPException(status_code=404, detail="Module not found or access denied")

    cursor.execute(""" DELETE FROM modules WHERE module_code = ? """, (module_code,))
    connection.commit()
    connection.close()
    return {"message": f"Module {module_code} deleted successfully"}




