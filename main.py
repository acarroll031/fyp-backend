import io
from contextlib import asynccontextmanager
from dataProcessing import convert_grades_to_students

import pandas as pd
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime, timedelta
from typing import Optional
import bcrypt
import jwt # For the tokens
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy import text

from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is not set")

    conn = psycopg2.connect(db_url)
    return conn

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
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    query="""
        SELECT s.student_id, s.student_name, s.module, s.risk_score
        FROM risk_scores s
        JOIN modules m ON s.module = m.module_code
        WHERE m.lecturer_email = %s
    """

    cursor.execute(query , (current_user_email,))
    rows = cursor.fetchall()
    connection.close()

    return rows


@app.post("/students/{module_id}/grades")
async def post_grades(
        module_id: str,
        progress_in_semester: float,
        file: UploadFile = File(...),
        model=Depends(get_model)
):
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    engine = create_engine(db_url)
    connection = engine.connect()

    raw_conn = get_db_connection()
    cursor = raw_conn.cursor()

    try:
        contents = await file.read()
        grades_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        grades_df["module"] = module_id
        grades_df["progress_in_semester"] = progress_in_semester

        grades_data = grades_df.to_dict(orient='records')

        upsert_query = """
                       INSERT INTO grades (student_id, student_name, assessment_number, score, module, 
                                           progress_in_semester)
                       VALUES (%(student_id)s, %(student_name)s, %(assessment_number)s, %(score)s, %(module)s, 
                               %(progress_in_semester)s) ON CONFLICT (student_id, module, assessment_number)
            DO 
                       UPDATE SET
                           score = EXCLUDED.score, 
                           progress_in_semester = EXCLUDED.progress_in_semester, 
                           student_name = EXCLUDED.student_name; 
                       """
        cursor.executemany(upsert_query, grades_data)
        raw_conn.commit()

        total_grades_df = pd.read_sql_query(
            "SELECT * FROM grades WHERE module = %s",
            engine,
            params=(module_id,)
        )

        student_df = convert_grades_to_students(total_grades_df)

        student_data = student_df.to_dict(orient='records')

        upsert_query = """
                       INSERT INTO students (student_id, student_name, module, average_score, assessments_completed, 
                                             performance_trend, max_consecutive_misses, progress_in_semester)
                       VALUES (%(student_id)s, %(student_name)s, %(module)s, %(average_score)s, %(assessments_completed)s, 
                               %(performance_trend)s, %(max_consecutive_misses)s, %(progress_in_semester)s)
            ON CONFLICT (student_id)
            DO 
                       UPDATE SET
                           module = EXCLUDED.module,
                           average_score = EXCLUDED.average_score, 
                           assessments_completed = EXCLUDED.assessments_completed,
                           performance_trend = EXCLUDED.performance_trend,
                           max_consecutive_misses = EXCLUDED.max_consecutive_misses,
                           progress_in_semester = EXCLUDED.progress_in_semester,
                           student_name = EXCLUDED.student_name;
                       """
        cursor.executemany(upsert_query, student_data)
        raw_conn.commit()

        features = student_df[[
            "average_score",
            "assessments_completed",
            "performance_trend",
            "progress_in_semester",
            "max_consecutive_misses"
        ]]

        risk_scores = model.predict(features)
        student_df["risk_score"] = risk_scores

        risk_scores_df = student_df[[
            "student_id",
            "student_name",
            "module",
            "risk_score"
        ]].copy()
        risk_scores_df['risk_score'] = risk_scores_df['risk_score'].round(2)

        risk_scores_data = risk_scores_df.to_dict(orient='records')

        upsert_query = """
                       INSERT INTO risk_scores (student_id, student_name, module, risk_score)
                       VALUES (%(student_id)s, %(student_name)s, %(module)s, %(risk_score)s)
            ON CONFLICT (student_id, module)
            DO 
                       UPDATE SET
                           risk_score = EXCLUDED.risk_score,
                           student_name = EXCLUDED.student_name;
                       """
        cursor.executemany(upsert_query, risk_scores_data)
        raw_conn.commit()

        return {"message": "Grades inserted and risk scores updated successfully"}

    except Exception as e:
        raw_conn.rollback()
        raise e
    finally:
        connection.close()
        cursor.close()
        raw_conn.close()

class LecturerCreate(BaseModel):
    email: str
    password: str
    lecturer_name: str

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/register")
def register_lecturer(lecturer: LecturerCreate):
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    hashed_password = get_password_hash(lecturer.password)

    try:
        cursor.execute("""
            INSERT INTO lecturers (email, password_hash, lecturer_name)
            VALUES (%s,%s ,%s)
        """, (lecturer.email, hashed_password, lecturer.lecturer_name)
        )
        connection.commit()
    except psycopg2.IntegrityError:
        connection.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        connection.close()

    return {"message": "Lecturer registered successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    cursor.execute(""" SELECT * FROM lecturers WHERE email = %s""", (form_data.username, ))
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
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute("""
            INSERT INTO modules (module_name, module_code, assessment_count, lecturer_email) VALUES (%s, %s, %s, %s)
        """,
         (module.module_name, module.module_code, module.assessment_count, current_user))
        connection.commit()
    except psycopg2.IntegrityError:
        connection.rollback()
        raise HTTPException(status_code=400, detail="Module code already exists")
    finally:
        connection.close()

    return {"message": f"Module {module.module_code} created successfully"}

@app.get("/modules")
def get_modules(
        current_user_email: str = Depends(get_current_user)
):
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE lecturer_email = %s
    """, (current_user_email,))
    modules = cursor.fetchall()
    connection.close()

    return modules

@app.put("/modules/{module_code}")
def update_module(
        module_code: str,
        module_update: ModuleUpdate,
        current_user_email: str = Depends(get_current_user)
):
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE module_code = ? AND lecturer_email = %s
    """, (module_code, current_user_email))
    if not cursor.fetchone():
        connection.close()
        raise HTTPException(status_code=404, detail="Module not found or access denied")

    if module_update.module_name:
        cursor.execute(""" UPDATE modules SET module_name = ? WHERE module_code = %s """, (module_update.module_name, module_code))
    if module_update.assessment_count:
        cursor.execute(""" UPDATE modules SET assessment_count = ? WHERE module_code = %s """, (module_update.assessment_count, module_code))

    connection.commit()
    connection.close()
    return {"message": f"Module {module_update.module_code} updated successfully"}

@app.delete("/modules/{module_code}")
def delete_module(
        module_code: str,
        current_user_email: str = Depends(get_current_user)
):
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE module_code = %s AND lecturer_email = %s
    """, (module_code, current_user_email))
    if not cursor.fetchone():
        connection.close()
        raise HTTPException(status_code=404, detail="Module not found or access denied")

    cursor.execute(""" DELETE FROM modules WHERE module_code = %s """, (module_code,))
    connection.commit()
    connection.close()
    return {"message": f"Module {module_code} deleted successfully"}




