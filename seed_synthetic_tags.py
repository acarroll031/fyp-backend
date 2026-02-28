"""
Seed Synthetic Data for Student Risk Tags

This script seeds the cloud database with synthetic student data to provide examples
for each of the different risk tags a student can have:

Tag Logic:
- "Newly At Risk": risk_score > 70 AND previous_risk_score <= 70
- "At Risk": risk_score > 70 AND previous_risk_score > 70
- "Improving": risk_score > 20 AND risk_score < previous_risk_score AND risk_score <= 70
- "On Track": risk_score <= 20 OR (risk_score <= 70 AND risk_score >= previous_risk_score)
"""

import os
import random
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODULE_CODE = "CS123"
LECTURER_EMAIL = "demo_user@test.com"
TOTAL_ASSESSMENTS = 12  # Total assessments in the semester
CURRENT_ASSESSMENT_COUNT = 8  # How many assessments have been uploaded so far (all students have this many)


def get_db_connection():
    """Get database connection using environment variable."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is not set")
    conn = psycopg2.connect(db_url)
    return conn


def generate_email(student_id):
    """Generate a generic email that won't accidentally send to anyone."""
    return f"student{student_id}@example.invalid"


# Define test data for each tag category with realistic names
# All students have the same number of assessments (CURRENT_ASSESSMENT_COUNT)
# progress_in_semester is calculated as CURRENT_ASSESSMENT_COUNT / TOTAL_ASSESSMENTS
PROGRESS = round(CURRENT_ASSESSMENT_COUNT / TOTAL_ASSESSMENTS, 2)

SYNTHETIC_STUDENTS = [
    # "Newly At Risk" - risk_score > 70 AND previous_risk_score <= 70
    {
        "student_id": 90001,
        "student_name": "Emma Thompson",
        "module": MODULE_CODE,
        "risk_score": 75.0,
        "previous_risk_score": 50.0,
        "average_score": 45.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": -15.0,
        "max_consecutive_misses": 2,
        "progress_in_semester": PROGRESS,
        "tag": "Newly At Risk"
    },
    {
        "student_id": 90002,
        "student_name": "James Wilson",
        "module": MODULE_CODE,
        "risk_score": 85.0,
        "previous_risk_score": 65.0,
        "average_score": 35.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": -20.0,
        "max_consecutive_misses": 3,
        "progress_in_semester": PROGRESS,
        "tag": "Newly At Risk"
    },

    # "At Risk" - risk_score > 70 AND previous_risk_score > 70
    {
        "student_id": 90003,
        "student_name": "Oliver Brown",
        "module": MODULE_CODE,
        "risk_score": 80.0,
        "previous_risk_score": 75.0,
        "average_score": 30.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": -18.0,
        "max_consecutive_misses": 4,
        "progress_in_semester": PROGRESS,
        "tag": "At Risk"
    },
    {
        "student_id": 90004,
        "student_name": "Sophia Martinez",
        "module": MODULE_CODE,
        "risk_score": 90.0,
        "previous_risk_score": 85.0,
        "average_score": 25.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": -25.0,
        "max_consecutive_misses": 5,
        "progress_in_semester": PROGRESS,
        "tag": "At Risk"
    },

    # "Improving" - risk_score > 20 AND risk_score < previous_risk_score AND risk_score <= 70
    {
        "student_id": 90005,
        "student_name": "Liam O'Connor",
        "module": MODULE_CODE,
        "risk_score": 45.0,
        "previous_risk_score": 65.0,
        "average_score": 60.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": 12.0,
        "max_consecutive_misses": 1,
        "progress_in_semester": PROGRESS,
        "tag": "Improving"
    },
    {
        "student_id": 90006,
        "student_name": "Ava Chen",
        "module": MODULE_CODE,
        "risk_score": 30.0,
        "previous_risk_score": 50.0,
        "average_score": 70.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": 18.0,
        "max_consecutive_misses": 0,
        "progress_in_semester": PROGRESS,
        "tag": "Improving"
    },

    # "On Track" - risk_score <= 20 OR (risk_score <= 70 AND risk_score >= previous_risk_score)
    {
        "student_id": 90007,
        "student_name": "Noah Murphy",
        "module": MODULE_CODE,
        "risk_score": 15.0,
        "previous_risk_score": 10.0,
        "average_score": 85.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": 5.0,
        "max_consecutive_misses": 0,
        "progress_in_semester": PROGRESS,
        "tag": "On Track (Low Risk)"
    },
    {
        "student_id": 90008,
        "student_name": "Isabella Walsh",
        "module": MODULE_CODE,
        "risk_score": 18.0,
        "previous_risk_score": 20.0,
        "average_score": 82.0,
        "assessments_completed": CURRENT_ASSESSMENT_COUNT,
        "performance_trend": 3.0,
        "max_consecutive_misses": 0,
        "progress_in_semester": PROGRESS,
        "tag": "On Track (Stable)"
    },
]

# Additional "normal" students to fill out the class roster
NORMAL_STUDENTS = [
    {"student_id": 90009, "student_name": "Ethan Kelly", "average_score": 72.0, "risk_score": 28.0, "previous_risk_score": 30.0, "trend": 4.0},
    {"student_id": 90010, "student_name": "Mia Fitzgerald", "average_score": 68.0, "risk_score": 32.0, "previous_risk_score": 35.0, "trend": 2.0},
    {"student_id": 90011, "student_name": "Lucas Ryan", "average_score": 75.0, "risk_score": 25.0, "previous_risk_score": 28.0, "trend": 5.0},
    {"student_id": 90012, "student_name": "Charlotte Doyle", "average_score": 78.0, "risk_score": 22.0, "previous_risk_score": 24.0, "trend": 6.0},
    {"student_id": 90013, "student_name": "Mason Burke", "average_score": 65.0, "risk_score": 35.0, "previous_risk_score": 38.0, "trend": 1.0},
    {"student_id": 90014, "student_name": "Amelia Nolan", "average_score": 80.0, "risk_score": 20.0, "previous_risk_score": 22.0, "trend": 7.0},
    {"student_id": 90015, "student_name": "Benjamin Hayes", "average_score": 70.0, "risk_score": 30.0, "previous_risk_score": 32.0, "trend": 3.0},
    {"student_id": 90016, "student_name": "Harper Quinn", "average_score": 74.0, "risk_score": 26.0, "previous_risk_score": 28.0, "trend": 4.0},
    {"student_id": 90017, "student_name": "William Gallagher", "average_score": 67.0, "risk_score": 33.0, "previous_risk_score": 35.0, "trend": 2.0},
    {"student_id": 90018, "student_name": "Evelyn Moore", "average_score": 76.0, "risk_score": 24.0, "previous_risk_score": 26.0, "trend": 5.0},
    {"student_id": 90019, "student_name": "Alexander Brennan", "average_score": 71.0, "risk_score": 29.0, "previous_risk_score": 31.0, "trend": 3.0},
    {"student_id": 90020, "student_name": "Abigail Duffy", "average_score": 79.0, "risk_score": 21.0, "previous_risk_score": 23.0, "trend": 6.0},
]

# Convert normal students to full format - all use same assessment count
for student in NORMAL_STUDENTS:
    student["module"] = MODULE_CODE
    student["assessments_completed"] = CURRENT_ASSESSMENT_COUNT
    student["performance_trend"] = student.pop("trend")
    student["max_consecutive_misses"] = 0
    student["progress_in_semester"] = PROGRESS
    student["tag"] = "On Track"

# Combine all students
ALL_STUDENTS = SYNTHETIC_STUDENTS + NORMAL_STUDENTS


def verify_module_exists(cursor):
    """Verify that the module and lecturer exist."""

    # Check if lecturer exists
    cursor.execute("SELECT email FROM lecturers WHERE email = %s", (LECTURER_EMAIL,))
    if not cursor.fetchone():
        print(f"⚠️  Warning: Lecturer {LECTURER_EMAIL} not found in database")
        return False
    else:
        print(f"✅ Found lecturer: {LECTURER_EMAIL}")

    # Check if module exists
    cursor.execute("SELECT module_code FROM modules WHERE module_code = %s", (MODULE_CODE,))
    if not cursor.fetchone():
        print(f"⚠️  Warning: Module {MODULE_CODE} not found in database")
        return False
    else:
        print(f"✅ Found module: {MODULE_CODE}")

    return True


def seed_students(cursor):
    """Insert synthetic students into the students table."""

    for student in ALL_STUDENTS:
        # Generate generic email
        email = generate_email(student['student_id'])

        # Insert or update student record
        cursor.execute("""
            INSERT INTO students (student_id, student_name, email, module, average_score, 
                                  assessments_completed, performance_trend, max_consecutive_misses, 
                                  progress_in_semester)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (student_id, module)
            DO UPDATE SET
                student_name = EXCLUDED.student_name,
                email = EXCLUDED.email,
                average_score = EXCLUDED.average_score,
                assessments_completed = EXCLUDED.assessments_completed,
                performance_trend = EXCLUDED.performance_trend,
                max_consecutive_misses = EXCLUDED.max_consecutive_misses,
                progress_in_semester = EXCLUDED.progress_in_semester;
        """, (
            student['student_id'],
            student['student_name'],
            email,
            student['module'],
            student['average_score'],
            student['assessments_completed'],
            student['performance_trend'],
            student['max_consecutive_misses'],
            student['progress_in_semester']
        ))
        print(f"✅ Inserted student: {student['student_name']} ({student['tag']})")


def seed_risk_scores(cursor):
    """Insert synthetic risk scores into the risk_scores table."""

    for student in ALL_STUDENTS:
        # Insert or update risk score
        cursor.execute("""
            INSERT INTO risk_scores (student_id, student_name, module, risk_score, previous_risk_score)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (student_id, module)
            DO UPDATE SET
                student_name = EXCLUDED.student_name,
                risk_score = EXCLUDED.risk_score,
                previous_risk_score = EXCLUDED.previous_risk_score;
        """, (
            student['student_id'],
            student['student_name'],
            student['module'],
            student['risk_score'],
            student['previous_risk_score']
        ))
        print(f"✅ Inserted risk score for: {student['student_name']} "
              f"(risk: {student['risk_score']}, prev: {student['previous_risk_score']})")


def seed_risk_history(cursor):
    """
    Insert synthetic risk history records with timestamps spaced a week apart.
    All students get CURRENT_ASSESSMENT_COUNT entries in their risk history.
    """

    # Start date - go back from now based on how many assessments have been done
    base_date = datetime.now() - timedelta(weeks=CURRENT_ASSESSMENT_COUNT - 1)

    for student in ALL_STUDENTS:
        # Generate a trajectory of risk scores over time
        current_risk = student['risk_score']
        prev_risk = student['previous_risk_score']

        # Calculate risk scores for each week leading up to current state
        risk_history = []

        # Linear interpolation from an initial score to previous, then to current
        initial_score = prev_risk + (prev_risk - current_risk) * 0.5  # Extrapolate back
        initial_score = max(0, min(100, initial_score))  # Clamp to valid range

        for i in range(CURRENT_ASSESSMENT_COUNT):
            if i < CURRENT_ASSESSMENT_COUNT - 1:
                # Interpolate from initial to previous risk score
                progress = i / (CURRENT_ASSESSMENT_COUNT - 1)
                score = initial_score + (current_risk - initial_score) * progress
            else:
                # Last entry is the current risk score
                score = current_risk

            # Add some small random variation (except for the final score)
            if i < CURRENT_ASSESSMENT_COUNT - 1:
                score = max(0, min(100, score + random.uniform(-3, 3)))
            risk_history.append(round(score, 2))

        # Insert each historical risk score with a timestamp
        for i, score in enumerate(risk_history):
            timestamp = base_date + timedelta(weeks=i)

            cursor.execute("""
                INSERT INTO risk_history (student_id, student_name, module, risk_score, recorded_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                student['student_id'],
                student['student_name'],
                student['module'],
                score,
                timestamp
            ))

        print(f"✅ Inserted {CURRENT_ASSESSMENT_COUNT} risk history entries for: {student['student_name']}")


def seed_sample_grades(cursor):
    """
    Insert sample grades for all students.
    All students get the same number of assessments (CURRENT_ASSESSMENT_COUNT).
    Zeros are regular scores (student got 0 on that assessment), not "missed" assessments.
    progress_in_semester = assessment_number / TOTAL_ASSESSMENTS
    """

    for student in ALL_STUDENTS:
        avg_score = student['average_score']
        trend = student['performance_trend']
        max_zeros = student['max_consecutive_misses']  # Number of zeros to include in scores

        # Generate scores that average to avg_score and show the trend
        scores = []

        # Determine which assessments should be zeros (for at-risk students)
        # Place zeros towards the end to show declining performance
        zero_positions = set()
        if max_zeros > 0:
            # Put zeros at the end of the assessment list
            for z in range(max_zeros):
                zero_positions.add(CURRENT_ASSESSMENT_COUNT - 1 - z)

        for i in range(CURRENT_ASSESSMENT_COUNT):
            if i in zero_positions:
                # This is a zero score (student got 0, but it's still a submitted grade)
                score = 0
            else:
                # Calculate base score with trend effect
                # If trend is positive, scores increase over time; if negative, they decrease
                trend_effect = (trend / CURRENT_ASSESSMENT_COUNT) * (i + 1)
                base_score = avg_score - (trend / 2) + trend_effect

                # Add some random variance
                score = base_score + random.uniform(-8, 8)
                score = max(0, min(100, score))

            scores.append(round(score, 2))

        # Insert each grade with progress_in_semester = assessment_number / TOTAL_ASSESSMENTS
        for assessment_num, score in enumerate(scores, 1):
            progress = round(assessment_num / TOTAL_ASSESSMENTS, 2)  # e.g., 1/12 = 0.08, 8/12 = 0.67

            cursor.execute("""
                INSERT INTO grades (student_id, student_name, module, assessment_number, score, progress_in_semester)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (student_id, module, assessment_number)
                DO UPDATE SET
                    score = EXCLUDED.score,
                    student_name = EXCLUDED.student_name,
                    progress_in_semester = EXCLUDED.progress_in_semester;
            """, (
                student['student_id'],
                student['student_name'],
                student['module'],
                assessment_num,
                score,
                progress
            ))

        print(f"✅ Inserted {CURRENT_ASSESSMENT_COUNT} grades for: {student['student_name']}")


def main():
    """Main function to seed all synthetic data."""
    print("\n🚀 Starting synthetic data seeding for risk tag examples...\n")

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Step 1: Verify module and lecturer exist
        print("=" * 50)
        print("Step 1: Verifying module and lecturer exist")
        print("=" * 50)
        if not verify_module_exists(cursor):
            print("\n⚠️  Please ensure the module and lecturer exist before seeding.")
            return

        # Step 2: Insert students
        print("\n" + "=" * 50)
        print("Step 2: Inserting synthetic students")
        print("=" * 50)
        seed_students(cursor)
        connection.commit()

        # Step 3: Insert risk scores
        print("\n" + "=" * 50)
        print("Step 3: Inserting risk scores")
        print("=" * 50)
        seed_risk_scores(cursor)
        connection.commit()

        # Step 4: Insert risk history
        print("\n" + "=" * 50)
        print("Step 4: Inserting risk history (weekly timestamps)")
        print("=" * 50)
        seed_risk_history(cursor)
        connection.commit()

        # Step 5: Insert sample grades
        print("\n" + "=" * 50)
        print("Step 5: Inserting sample grades")
        print("=" * 50)
        seed_sample_grades(cursor)
        connection.commit()

        print("\n" + "=" * 50)
        print("✨ Synthetic data seeding complete!")
        print("=" * 50)
        print("\nSummary of seeded students by tag:")
        print("-" * 50)
        for student in SYNTHETIC_STUDENTS:
            print(f"  • {student['student_name']}: {student['tag']}")
        print("-" * 50)
        print(f"\nAdditional 'normal' students added: {len(NORMAL_STUDENTS)}")
        print(f"Total students seeded: {len(ALL_STUDENTS)}")
        print(f"\nModule: {MODULE_CODE}")
        print(f"Lecturer: {LECTURER_EMAIL}")

    except Exception as e:
        connection.rollback()
        print(f"\n❌ Error during seeding: {e}")
        raise
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()

