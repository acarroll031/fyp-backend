import os
import psycopg2
from urllib.parse import urlparse
import csv

#TODO: Reinsert sample grade manually to fix progress_in_semester values (were set to int instead of real, so no decimal values)

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


    # Add this helper function after execute_sql_file()
    def load_csv_to_table(csv_filename, table_name, columns):
        """
        Load data from CSV file into a database table.

        Args:
            csv_filename: Path to the CSV file
            table_name: Name of the table to insert into
            columns: List of column names in the table
        """
        print(f"--- Loading data into {table_name} from {csv_filename} ---")
        try:
            with open(csv_filename, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                rows_inserted = 0

                for row in csv_reader:
                    # Build INSERT query dynamically
                    placeholders = ', '.join(['%s'] * len(columns))
                    column_names = ', '.join(columns)
                    insert_query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

                    # Extract values in the correct order
                    values = [row.get(col) for col in columns]

                    try:
                        cursor_obj.execute(insert_query, values)
                        rows_inserted += 1
                    except Exception as msg:
                        print(f"❌ ERROR inserting row into {table_name}: {msg}")
                        print(f"Failed row data: {row}")
                        connection_obj.rollback()
                        return

                connection_obj.commit()
                print(f"✅ Loaded {rows_inserted} rows into {table_name}")

        except FileNotFoundError:
            print(f"Error: CSV file {csv_filename} not found.")
        except Exception as e:
            print(f"Error loading CSV: {e}")


    # 5. Run the scripts
    # 1. Lecturers (Must be first - Parents)
    execute_sql_file('SQL Scripts/lecturers.sql', 'Lecturers')

    # 2. Modules (Depends on Lecturers)
    execute_sql_file('SQL Scripts/modules.sql', 'Modules')

    # 3. Students (Depends on Modules)
    execute_sql_file('SQL Scripts/students.sql', 'Students')

    load_csv_to_table(
        'CSV Files/students.csv',
        'Students',
        ['student_id', 'student_name', 'module', 'average_score', 'assessments_completed', 'performance_trend', 'max_consecutive_misses', 'progress_in_semester']
    )

    # 4. Grades (Depends on Students & Modules)
    execute_sql_file('SQL Scripts/grades.sql', 'Grades')

    load_csv_to_table(
        'CSV Files/grades.csv',
        'Grades',
        ['student_id','student_name', 'module','assessment_number', 'score', 'progress_in_semester']
    )

    # 5. Risk Scores (Depends on Students)
    execute_sql_file('SQL Scripts/risk_scores.sql', 'Risk Scores')

    load_csv_to_table(
        'CSV Files/risk_scores.csv',
        'Risk_Scores',
        ['student_id', 'student_name', 'module', 'risk_score']
    )

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