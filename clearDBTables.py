import sqlite3

connection_obj = sqlite3.connect('fyp_database.db')
cursor_obj = connection_obj.cursor()

clear_grade_table = '''
    DELETE FROM grades;
'''

clear_student_table = '''
    DELETE FROM students;
'''

cursor_obj.execute(clear_grade_table)
print("Grades table cleared successfully")
cursor_obj.execute(clear_student_table)
print("Students table cleared successfully")

connection_obj.commit()
connection_obj.close()