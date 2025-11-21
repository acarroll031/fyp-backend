import sqlite3

connection_obj = sqlite3.connect('fyp_database.db')
cursor_obj = connection_obj.cursor()

cursor_obj.execute("DROP TABLE IF EXISTS grades")
cursor_obj.execute("DROP TABLE IF EXISTS students")
cursor_obj.execute("DROP TABLE IF EXISTS risk_scores")
cursor_obj.execute("DROP TABLE IF EXISTS lecturers")
cursor_obj.execute("DROP TABLE IF EXISTS modules")

# Create Grades Table
fd = open('SQL Scripts/grades.sql', 'r')
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(';')
for command in sqlCommands:
    try:
        cursor_obj.execute(command)
    except sqlite3.OperationalError as msg:
        print("Command skipped: ", msg)

print("Grades table created successfully")


# Create Students Table
fd = open('SQL Scripts/students.sql', 'r')
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(';')
for command in sqlCommands:
    try:
        cursor_obj.execute(command)
    except sqlite3.OperationalError as msg:
        print("Command skipped: ", msg)

print("Students table created successfully")

# Create Risk Scores Table
fd = open('SQL Scripts/risk_scores.sql', 'r')
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(';')
for command in sqlCommands:
    try:
        cursor_obj.execute(command)
    except sqlite3.OperationalError as msg:
        print("Command skipped: ", msg)

print("Risk Scores table created successfully")

# Create Lecturers Table
fd = open('SQL Scripts/lecturers.sql', 'r')
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(';')
for command in sqlCommands:
    try:
        cursor_obj.execute(command)
    except sqlite3.OperationalError as msg:
        print("Command skipped: ", msg)

print("Lecturers table created successfully")

# Create Modules Table
fd = open('SQL Scripts/modules.sql', 'r')
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(';')
for command in sqlCommands:
    try:
        cursor_obj.execute(command)
    except sqlite3.OperationalError as msg:
        print("Command skipped: ", msg)

print("Modules table created successfully")


connection_obj.commit()
connection_obj.close()

