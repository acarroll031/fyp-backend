import sqlite3

connection_obj = sqlite3.connect('fyp_database.db')
cursor_obj = connection_obj.cursor()

create_grade_table = '''
    CREATE TABLE IF NOT EXISTS grades (
        student_id INTEGER PRIMARY KEY,
        student_name TEXT NOT NULL,
        module TEXT NOT NULL,
        assessment_number INTEGER NOT NULL,
        score REAL NOT NULL,
        progress_in_semester INTEGER NOT NULL
    );
'''

create_student_table = '''
    CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY,
        student_name TEXT NOT NULL,
        module TEXT NOT NULL,
        average_score REAL,
        assessments_completed INTEGER,
        performance_trend REAL,
        max_consecutive_misses INTEGER,
        progress_in_semester FLOAT
    );'''

create_risk_score_table = '''
    CREATE TABLE IF NOT EXISTS risk_scores (
        student_id INTEGER PRIMARY KEY,
        student_name TEXT NOT NULL,
        module TEXT NOT NULL,
        risk_score REAL
    );'''
cursor_obj.execute("DROP TABLE IF EXISTS grades")
cursor_obj.execute("DROP TABLE IF EXISTS students")
cursor_obj.execute("DROP TABLE IF EXISTS risk_scores")

cursor_obj.execute(create_grade_table)
print("Grades table created successfully")
cursor_obj.execute(create_student_table)
print("Students table created successfully")
cursor_obj.execute(create_risk_score_table)
print("Risk Scores table created successfully")


cursor_obj.execute("INSERT INTO risk_scores (student_id, student_name, module, risk_score) VALUES (1, 'Adam', 'CS161', 45)")
cursor_obj.execute("INSERT INTO risk_scores (student_id, student_name, module, risk_score) VALUES (2, 'Daniel', 'CS161', 5)")
cursor_obj.execute("INSERT INTO risk_scores (student_id, student_name, module, risk_score) VALUES (3, 'Luke', 'CS162', 90.0)")

print("Data Inserted in the table: ")
cursor_obj.execute("SELECT * FROM risk_scores")
print(cursor_obj.fetchall())

connection_obj.commit()
connection_obj.close()

