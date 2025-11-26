import os
import psycopg2
from urllib.parse import urlparse

# 1. Get the DB URL from your environment (set this in your IDE or terminal)
# Example: postgres://user:password@ep-cool-frog.eu-central-1.aws.neon.tech/neondb
DATABASE_URL = "postgresql://neondb_owner:npg_GzHPrUVFJ21u@ep-crimson-glade-abkp5rc9-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Please set it as an environment variable.")

try:
    # 2. Connect to the PostgreSQL Database
    connection_obj = psycopg2.connect(DATABASE_URL)
    cursor_obj = connection_obj.cursor()
    print("Successfully connected to PostgreSQL")

    # 3. Drop Tables (Use CASCADE to handle Foreign Keys automatically)
    tables_to_drop = ["grades", "students", "risk_scores", "lecturers", "modules"]
    for table in tables_to_drop:
        cursor_obj.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
        print(f"Dropped table: {table}")


    # 4. Helper function to execute SQL files
    def execute_sql_file(filename, table_name):
        print(f"--- Processing {table_name} ---")
        try:
            with open(filename, 'r') as fd:
                sqlFile = fd.read()

            sqlCommands = sqlFile.split(';')

            for command in sqlCommands:
                if command.strip():
                    try:
                        cursor_obj.execute(command)
                    except Exception as msg:
                        print(f"❌ ERROR in {table_name}: {msg}")
                        print(f"Failed Command: {command[:100]}...")  # Print start of bad command
                        connection_obj.rollback()  # Reset the transaction so we can see the next error
                        return  # Stop this file

            print(f"✅ {table_name} table processed successfully")
            connection_obj.commit()  # Save progress after each table

        except FileNotFoundError:
            print(f"Error: File {filename} not found.")

    # 5. Run the scripts
    # 1. Lecturers (Must be first - Parents)
    execute_sql_file('SQL Scripts/lecturers.sql', 'Lecturers')

    # 2. Modules (Depends on Lecturers)
    execute_sql_file('SQL Scripts/modules.sql', 'Modules')

    # 3. Students (Depends on Modules)
    execute_sql_file('SQL Scripts/students.sql', 'Students')

    # 4. Grades (Depends on Students & Modules)
    execute_sql_file('SQL Scripts/grades.sql', 'Grades')

    # 5. Risk Scores (Depends on Students)
    execute_sql_file('SQL Scripts/risk_scores.sql', 'Risk Scores')

    # 6. Commit and Close
    connection_obj.commit()
    print("All operations committed successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    if connection_obj:
        connection_obj.rollback()  # Rollback changes on error

finally:
    if connection_obj:
        cursor_obj.close()
        connection_obj.close()
        print("Connection closed.")