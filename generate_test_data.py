import pandas as pd
import numpy as np
import random


def generate_test_csv(filename="test_upload2.csv", num_students=100, num_assessments=1):
    """
    Generates a dummy CSV file in the 'long' format for testing the
    student data upload endpoint.
    """

    # 1. Create lists of dummy names to pull from
    first_names = [
        'Aaliyah', 'Aaron', 'Abdula', 'Adam', 'Alex', 'Megan', 'Michael',
        'Leo', 'Killian', 'Kate', 'Ben', 'Ciara', 'Conor', 'Darragh'
    ]
    last_names = [
        'Doyle', 'Moran', 'Kennedy', 'Sheehan', 'Boyle', 'Smith', 'Quinn',
        'McMahon', 'McCarthy', 'Ryan', 'Murphy', 'Walsh', 'Kelly', 'Byrne'
    ]

    data = []

    # 2. Create 10 unique students
    for i in range(num_students):
        student_id = 1000 + i
        student_name = f"{random.choice(first_names)} {random.choice(last_names)}"

        # 3. For each student, create 10 assessment entries
        for j in range(2, num_assessments + 2):
            assessment_number = j

            # Simulate a realistic score (e.g., 70% chance of a good score, 30% chance of a low/zero score)
            if random.random() < 0.75:
                score = round(random.uniform(60, 100), 2)
            else:
                score = round(random.uniform(0, 50), 2)

            # Add the row to our data list
            data.append({
                "student_id": student_id,
                "student_name": student_name,
                "assessment_number": assessment_number,
                "score": score
            })

    # 4. Convert the list of data into a pandas DataFrame
    df = pd.DataFrame(data)

    # 5. Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

    print(f"Successfully generated '{filename}' with {len(df)} entries.")
    print("\n--- Sample of Generated Data ---")
    print(df.head())


# --- Run the function ---
if __name__ == "__main__":
    generate_test_csv()