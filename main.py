import io
import json
import os
import bcrypt
import joblib
import jwt  # For the tokens
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from sqlalchemy import create_engine
from dataProcessing import convert_grades_to_students
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

load_dotenv()

# Database connection
def get_db_connection():
    db_url = os.getenv("DATABASE_URL") # Get the database URL from environment variable
    if not db_url:
        raise ValueError("DATABASE_URL is not set") # Catch missing env variable

    conn = psycopg2.connect(db_url) # Connect to the PostgreSQL database
    return conn

SECRET_KEY = "secret" # In production, use a secure method to store this
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Setup OAuth2 for token handling

def get_password_hash(password: str) -> str:
    """
    Hash a password for storing.
    :param password: The plain password to hash
    :return: The hashed password
    """
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a stored password against one provided by user
    :param plain_password: The plain password to verify
    :param hashed_password: The stored hashed password
    :return: Boolean indicating if the password matches
    """
    password_byte_encoded = plain_password.encode('utf-8')
    hashed_byte_encoded = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte_encoded, hashed_byte_encoded)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.
    :param data: The data to encode in the token
    :param expires_delta: The time delta for token expiration
    :return: An encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Decode the JWT token to get the current user.
    :param token: The JWT token
    :return: The email of the current user
    """
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

# FastAPI app with lifespan event to load the model
@asynccontextmanager
async def lifespan(app: FastAPI):
    """" Load the ML model at startup and unload at shutdown """
    app.state.model = joblib.load("student_risk_model_0.1-1.0.joblib")
    yield
    app.state.model = None

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# CORS middleware setup
origins = ["*"]

# Allow CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    """ Dependency to get the ML model """
    return app.state.model

class PredictRequest(BaseModel):
    """ Input features for predicting student risk score """
    average_score: float
    assessments_completed: int
    performance_trend: float
    max_consecutive_misses: int
    progress_in_semester: float

@app.post("/predict", summary="Predict Student Risk Score", description="Predict the risk score of a student based on their performance features.")
def predict_risk(
        request: PredictRequest,
        model=Depends(get_model)
):
    """
    Predict the risk score for a student based on input features.
    :param request: PredictRequest containing student features
    :param model: The ML model for prediction
    :return: Predicted risk score
    """
    # Reshape input for model as 2D array as required by scikit-learn
    features = [[
        request.average_score,
        request.assessments_completed,
        request.performance_trend,
        request.max_consecutive_misses,
        request.progress_in_semester
    ]]
    risk_score = model.predict(features)
    return {"risk_score": risk_score[0]}

@app.get("/students", summary="Get Students for Lecturer", description="Retrieve a list of students associated with the logged-in lecturer.")
def get_students(current_user_email: str = Depends(get_current_user)):
    """
    Get a list of students for the logged-in lecturer.
    :param current_user_email: The email of the current logged-in lecturer
    :return: List of students with their risk scores
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    # SQL query to fetch students associated with the lecturer
    query="""
        SELECT s.student_id, s.student_name, s.module, s.risk_score, s.previous_risk_score
        FROM risk_scores s
        JOIN modules m ON s.module = m.module_code
        WHERE m.lecturer_email = %s
    """

    cursor.execute(query , (current_user_email,))
    rows = cursor.fetchall()
    connection.close()

    return rows

@app.get("/students/{student_id}/{module_id}", summary="Get Student Details by Module", description="Retrieve detailed information about a specific student for a specific module, including grades and risk history.")
def get_student_details_by_module(student_id: str, module_id: str, current_user_email: str = Depends(get_current_user)):
    """
    Get detailed information about a specific student for a specific module
    :param student_id: ID of the student
    :param module_id: ID of the module
    :param current_user_email: Email of the current logged-in lecturer
    :return: Student details, grades, and risk history as a JSON object with three keys: "student", "grades", and "risk_history"
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    try:
        # SQL query to fetch student details
        cursor.execute("""
                       SELECT s.*, r.risk_score
                       FROM students s
                        JOIN modules m ON s.module = m.module_code
                        JOIN risk_scores r ON s.student_id = r.student_id AND s.module = r.module
                       WHERE s.student_id = %s
                            AND s.module = %s
                         AND m.lecturer_email = %s
                       """, (student_id, module_id , current_user_email))

        student = cursor.fetchone()

        # If no student found, raise 404 error
        if not student:
            raise HTTPException(status_code=404, detail="Student not found or access denied")

        # SQL query to fetch grades
        cursor.execute("""
                       SELECT assessment_number, score, progress_in_semester
                       FROM grades
                       WHERE student_id = %s
                         AND module = %s
                       ORDER BY assessment_number
                       """, (student_id, module_id))

        grades = cursor.fetchall()

        # SQL query to fetch risk history
        cursor.execute("""
                       SELECT risk_score as risk_score_history, recorded_at as risk_score_history_timestamp
                       FROM risk_history
                       WHERE student_id = %s
                         AND module = %s
                       ORDER BY recorded_at ASC
                       """, (student_id, module_id))

        risk_history = cursor.fetchall()

        return {"student": student, "grades": grades, "risk_history": risk_history}

    finally:
        connection.close()

@app.post("/students/{module_id}/grades", summary="Upload Grades and Update Risk Scores", description="Upload a CSV file containing student grades for a specific module, update the grades in the database, and recalculate risk scores using the ML model.")
async def post_grades(
        module_id: str,
        progress_in_semester: float,
        file: UploadFile = File(...),
        model=Depends(get_model)
):
    """
    Upload grades CSV, update grades, and recalculate risk scores.
    :param module_id: ID of the module
    :param progress_in_semester: Value indicating progress in the semester from 0.0 to 1.0
    :param file: CSV file containing grades
    :param model: ML model for risk score prediction
    :return: Success message
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1) # SQLAlchemy requires 'postgresql://'

    engine = create_engine(db_url)
    connection = engine.connect()

    raw_conn = get_db_connection()
    cursor = raw_conn.cursor()

    try:
        contents = await file.read()
        grades_df = pd.read_csv(io.StringIO(contents.decode("utf-8"))) # Read CSV into DataFrame

        ### Grades Upsert Logic ###

        # Add module and progress columns
        grades_df["module"] = module_id
        grades_df["progress_in_semester"] = progress_in_semester
        # Create an email mapping to preserve emails to add back in later
        email_mapping = grades_df[["student_id", "email"]].drop_duplicates()
        grades_data = grades_df.to_dict(orient='records')

        # SQL Query to add students to the students table if they don't exist (without features, just ID, name, email, module)
        add_students = """
               INSERT INTO students (student_id, student_name, email, module)
               VALUES (%(student_id)s, %(student_name)s, %(email)s, %(module)s)
               ON CONFLICT (student_id, module) DO NOTHING
               """
        cursor.executemany(add_students, grades_data)
        raw_conn.commit()

        # SQL query to upsert grades
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

        # Fetch all grades for the module to recalculate student summaries
        total_grades_df = pd.read_sql_query(
            "SELECT * FROM grades WHERE module = %s",
            engine,
            params=(module_id,)
        )

        # Convert all grades to student summaries with features for ML model
        student_df = convert_grades_to_students(total_grades_df)

        # Add emails back in
        student_df = student_df.merge(email_mapping, on=["student_id"], how="left")

        # Convert student summaries to list of dicts for insertion
        student_data = student_df.to_dict(orient='records')

        # SQL query to upsert student summaries
        upsert_query = """
                       INSERT INTO students (student_id, student_name, email, module, average_score, assessments_completed, 
                                             performance_trend, max_consecutive_misses, progress_in_semester)
                       VALUES (%(student_id)s, %(student_name)s, %(email)s, %(module)s, %(average_score)s, %(assessments_completed)s, 
                               %(performance_trend)s, %(max_consecutive_misses)s, %(progress_in_semester)s)
            ON CONFLICT (student_id, module)
            DO 
                       UPDATE SET
                           average_score = EXCLUDED.average_score, 
                           assessments_completed = EXCLUDED.assessments_completed,
                           performance_trend = EXCLUDED.performance_trend,
                           max_consecutive_misses = EXCLUDED.max_consecutive_misses,
                           progress_in_semester = EXCLUDED.progress_in_semester,
                           student_name = EXCLUDED.student_name,
                           email = EXCLUDED.email;
                       """
        cursor.executemany(upsert_query, student_data)
        raw_conn.commit()

        ### Risk Score Calculation Logic ###

        # Ensure features have the correct name and order for the model
        features = student_df[[
            "average_score",
            "assessments_completed",
            "performance_trend",
            "progress_in_semester",
            "max_consecutive_misses"
        ]]

        # Fetch existing risk scores for comparison
        existing_scores_query = "SELECT student_id, risk_score FROM risk_scores WHERE module = %s"
        existing_scores_df = pd.read_sql_query(existing_scores_query, engine, params=(module_id,))

        # Rename column to avoid confusion
        existing_scores_df = existing_scores_df.rename(columns={"risk_score": "previous_risk_score"})

        # Predict new risk scores using the ML model
        risk_scores = model.predict(features)
        student_df["risk_score"] = risk_scores

        # Merge the previous scores into our main dataframe
        student_df = pd.merge(student_df, existing_scores_df, on="student_id", how="left")
        # Fill NaN (new students have no history) with 0 or the current score
        student_df["previous_risk_score"] = student_df["previous_risk_score"].fillna(0)

        # Create a new dataframe for inserting into risk_scores
        risk_scores_df = student_df[[
            "student_id",
            "student_name",
            "module",
            "risk_score",
            "previous_risk_score"
        ]].copy()
        # Round risk scores to 2 decimal places for readability
        risk_scores_df['risk_score'] = risk_scores_df['risk_score'].round(2)
        # Convert to list of dicts for insertion
        risk_scores_data = risk_scores_df.to_dict(orient='records')

        # Insert into risk_history table for tracking over time
        history_query = """
                        INSERT INTO risk_history (student_id, student_name, module, risk_score)
                        VALUES (%(student_id)s, %(student_name)s, %(module)s, %(risk_score)s) \
                        """
        cursor.executemany(history_query, risk_scores_data)
        raw_conn.commit()

        # Upsert into risk_scores table
        upsert_query = """
                       INSERT INTO risk_scores (student_id, student_name, module, risk_score, previous_risk_score)
                       VALUES (%(student_id)s, %(student_name)s, %(module)s, %(risk_score)s, %(previous_risk_score)s)
            ON CONFLICT (student_id, module)
            DO 
                       UPDATE SET
                           risk_score = EXCLUDED.risk_score,
                           student_name = EXCLUDED.student_name,
                           previous_risk_score = EXCLUDED.previous_risk_score;
                       """
        cursor.executemany(upsert_query, risk_scores_data)
        raw_conn.commit()

        ### Notification Logic ###

        # Identify students who have newly become at risk (risk_score > 70) and were not previously at risk (previous_risk_score <= 70)
        newly_at_risk_students = student_df[
            (student_df["risk_score"] > 70) &
            (student_df["previous_risk_score"] <= 70) &
            (student_df["previous_risk_score"] > 0)
        ]

        # Fetch lecturer email for the module
        cursor.execute("SELECT lecturer_email FROM modules WHERE module_code = %s", (module_id,))
        lecturer_email = cursor.fetchone()[0]

        # Insert notifications for newly at-risk students
        if not newly_at_risk_students.empty:

            if lecturer_email:
                notifications_data = []

                for _, row in newly_at_risk_students.iterrows():
                    payload = {
                        "student_id": row['student_id'],
                        "module_id": row['module'],
                        "text": f"Student {row['student_name']} (ID: {row['student_id']}) has become at risk with a score of {round(row['risk_score'], 2)}.",
                    }

                    notifications_data.append({
                        "lecturer_email": lecturer_email,
                        "message": json.dumps(payload),
                        "notification_type": "RISK_ALERT",
                        "module_id": module_id
                    })

                notification_query = """
                                     INSERT INTO notifications (lecturer_email, message, notification_type, module)
                                     VALUES (%(lecturer_email)s, %(message)s, %(notification_type)s, %(module_id)s) \
                                     """
                cursor.executemany(notification_query, notifications_data)
                raw_conn.commit()

        # Insert notification for grades uploaded successfully
        payload = {
            "module_id": module_id,
            "text": f"Grades for module {module_id} have been uploaded successfully and risk scores updated."
        }
        notification_data = {
            "lecturer_email": lecturer_email,
            "message": json.dumps(payload),
            "notification_type": "UPLOAD_SUCCESS",
            "module_id": module_id
        }

        notification_query = """
                             INSERT INTO notifications (lecturer_email, message, notification_type, module)
                             VALUES (%(lecturer_email)s, %(message)s, %(notification_type)s, %(module_id)s) \
                             """
        cursor.execute(notification_query, notification_data)
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
    """ Model for creating a new lecturer """
    email: str
    password: str
    lecturer_name: str

class Token(BaseModel):
    """ Model for JWT token response """
    access_token: str
    token_type: str

@app.post("/register", summary="Register Lecturer", description="Register a new lecturer with email, password, and name.")
def register_lecturer(lecturer: LecturerCreate):
    """
    Register a new lecturer.
    :param lecturer: LecturerCreate object containing email, password, and name
    :return: Success message
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    hashed_password = get_password_hash(lecturer.password)

    try:
        # Insert new lecturer into the database
        cursor.execute("""
            INSERT INTO lecturers (email, password_hash, lecturer_name)
            VALUES (%s,%s ,%s)
        """, (lecturer.email, hashed_password, lecturer.lecturer_name)
        )
        connection.commit()
    except psycopg2.IntegrityError:
        connection.rollback()
        raise HTTPException(status_code=400, detail="Email already registered") # Handle duplicate email
    finally:
        connection.close()

    return {"message": "Lecturer registered successfully"}

@app.post("/login", summary="Lecturer Login", description="Authenticate a lecturer and receive a JWT token.")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Lecturer login to receive JWT token.
    :param form_data: OAuth2PasswordRequestForm containing username and password
    :return: JWT token for authenticated lecturer
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    # Fetch lecturer by email
    cursor.execute(""" SELECT * FROM lecturers WHERE email = %s""", (form_data.username, ))
    lecturer = cursor.fetchone()
    connection.close()

    # Verify password
    if not lecturer or not verify_password(form_data.password, lecturer["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT token
    access_token = create_access_token(
        data={"sub": lecturer["email"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}

class ModuleCreate(BaseModel):
    """ Model for creating a new module """
    module_name: str
    module_code: str
    assessment_count: int

class ModuleUpdate(BaseModel):
    """ Model for updating module details """
    module_name: Optional[str] = None
    assessment_count: Optional[int] = None
    module_code: Optional[str] = None

@app.post("/modules", summary="Create Module", description="Create a new module associated with the logged-in lecturer.")
def create_module(
        module: ModuleCreate,
        current_user: str = Depends(get_current_user)
):
    """
    Create a new module.
    :param module: ID of the module to create
    :param current_user: Email of the current logged-in lecturer
    :return: Success message
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    try:
        # Insert new module into the database
        cursor.execute("""
            INSERT INTO modules (module_name, module_code, assessment_count, lecturer_email) VALUES (%s, %s, %s, %s)
        """,
         (module.module_name, module.module_code, module.assessment_count, current_user))
        connection.commit()
    except psycopg2.IntegrityError:
        connection.rollback()
        raise HTTPException(status_code=400, detail="Module code already exists") # Handle duplicate module code
    finally:
        connection.close()

    return {"message": f"Module {module.module_code} created successfully"}

@app.get("/modules", summary="Get Modules", description="Retrieve a list of modules associated with the logged-in lecturer.")
def get_modules(
        current_user_email: str = Depends(get_current_user)
):
    """
    Get modules for the logged-in lecturer.
    :param current_user_email: Email of the current logged-in lecturer
    :return: List of modules
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    # SQL query to fetch modules for the lecturer
    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE lecturer_email = %s
    """, (current_user_email,))
    modules = cursor.fetchall()
    connection.close()

    return modules

@app.put("/modules/{module_code}", summary="Update Module", description="Update details of a specific module associated with the logged-in lecturer.")
def update_module(
        module_code: str,
        module_update: ModuleUpdate,
        current_user_email: str = Depends(get_current_user)
):
    """
    Update a module's details.
    :param module_code: ID of the module to update
    :param module_update: ModuleUpdate object containing updated details
    :param current_user_email: Email of the current logged-in lecturer
    :return: Success message
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    # Check if module exists and belongs to the lecturer
    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE module_code = ? AND lecturer_email = %s
    """, (module_code, current_user_email))

    # If module not found, raise 404 error
    if not cursor.fetchone():
        connection.close()
        raise HTTPException(status_code=404, detail="Module not found or access denied")

    # Update module details
    if module_update.module_name:
        cursor.execute(""" UPDATE modules SET module_name = ? WHERE module_code = %s """, (module_update.module_name, module_code))
    if module_update.assessment_count:
        cursor.execute(""" UPDATE modules SET assessment_count = ? WHERE module_code = %s """, (module_update.assessment_count, module_code))

    connection.commit()
    connection.close()
    return {"message": f"Module {module_update.module_code} updated successfully"}

@app.delete("/modules/{module_code}", summary="Delete Module", description="Delete a specific module associated with the logged-in lecturer.")
def delete_module(
        module_code: str,
        current_user_email: str = Depends(get_current_user)
):
    """
    Delete a module.
    :param module_code: ID of the module to delete
    :param current_user_email: Email of the current logged-in lecturer
    :return: Success message
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    # Check if module exists and belongs to the lecturer
    cursor.execute("""
        SELECT * 
        FROM modules 
        WHERE module_code = %s AND lecturer_email = %s
    """, (module_code, current_user_email))

    # If module not found, raise 404 error
    if not cursor.fetchone():
        connection.close()
        raise HTTPException(status_code=404, detail="Module not found or access denied")

    # Delete the module
    cursor.execute(""" DELETE FROM modules WHERE module_code = %s """, (module_code,))

    connection.commit()
    connection.close()

    return {"message": f"Module {module_code} deleted successfully"}

@app.get("/notifications", summary="Get Notifications", description="Retrieve notifications for the logged-in lecturer about students at risk.")
def get_notifications(current_user: str = Depends(get_current_user)):
    """
    Get notifications for the logged-in lecturer about students at risk.
    :param current_user: Email of the current logged-in lecturer
    :return: List of notifications
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    try:
        # SQL query to fetch notifications for the lecturer
        cursor.execute("""
            SELECT * FROM notifications
            WHERE lecturer_email = %s
            ORDER BY created_at DESC
        """, (current_user,))
        notifications = cursor.fetchall()
        return notifications
    finally:
        connection.close()

@app.put("/notifications/{notification_id}/unread", summary="Mark Notification as Unread", description="Mark a specific notification as unread for the logged-in lecturer.")
def mark_notification_as_read(notification_id: int, current_user: str = Depends(get_current_user)):
    """
    Mark a notification as unread.
    :param notification_id: ID of the notification to mark as unread
    :param current_user: Email of the current logged-in lecturer
    :return: Success message
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    try:
        # Check if notification exists and belongs to the lecturer
        cursor.execute("""
            SELECT * FROM notifications
            WHERE id = %s AND lecturer_email = %s
        """, (notification_id, current_user))

        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Notification not found or access denied")

        # Mark the notification as unread
        cursor.execute("""
            UPDATE notifications
            SET is_read = FALSE
            WHERE id = %s AND lecturer_email = %s
        """, (notification_id, current_user))

        connection.commit()
        return {"message": "Notification marked as unread"}
    finally:
        connection.close()

@app.put("/notifications/{notification_id}/read", summary="Mark Notification as Read", description="Mark a specific notification as read for the logged-in lecturer.")
def mark_notification_as_read(notification_id: int, current_user: str = Depends(get_current_user)):
    """
    Mark a notification as read.
    :param notification_id: ID of the notification to mark as read
    :param current_user: Email of the current logged-in lecturer
    :return: Success message
    """
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)

    try:
        # Check if notification exists and belongs to the lecturer
        cursor.execute("""
                       SELECT *
                       FROM notifications
                       WHERE id = %s
                         AND lecturer_email = %s
                       """, (notification_id, current_user))

        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Notification not found or access denied")

        # Mark the notification as read
        cursor.execute("""
                       UPDATE notifications
                       SET is_read = TRUE
                       WHERE id = %s
                         AND lecturer_email = %s
                       """, (notification_id, current_user))

        connection.commit()
        return {"message": "Notification marked as read"}
    finally:
        connection.close()

