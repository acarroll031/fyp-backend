import pandas as pd
import numpy as np
import os


def generate_full_student_data(num_students, num_assessments, module_id):
    """
    Generates a realistic DataFrame of dummy student data, including
    final outcome columns for calculating a risk score.
    """
    data = []
    assessment_columns = [f"Lab {i + 1}" for i in range(num_assessments)]

    for _ in range(num_students):
        student_id = np.random.randint(10000000, 99999999)
        archetype = np.random.choice(['high', 'declining', 'improving', 'average', 'disengaged'])

        # --- 1. Generate Assessment Scores based on Archetype ---
        if archetype == 'high':
            base_scores = np.random.normal(loc=90, scale=5, size=num_assessments)
        elif archetype == 'declining':
            base_scores = np.linspace(95, 40, num_assessments) + np.random.normal(0, 5, num_assessments)
        elif archetype == 'improving':
            base_scores = np.linspace(40, 95, num_assessments) + np.random.normal(0, 5, num_assessments)
        elif archetype == 'disengaged':
            # High probability of getting zero on any given assessment
            submitted_scores = np.random.normal(loc=65, scale=10, size=num_assessments)
            is_submitted = np.random.choice([1, 0], size=num_assessments, p=[0.3, 0.7])  # Only 30% submission rate
            base_scores = submitted_scores * is_submitted
        else:  # average
            base_scores = np.random.normal(loc=65, scale=15, size=num_assessments)

        scores = np.clip(base_scores, 0, 100).round(2)

        # # --- 2. Calculate Final Grade based on TOTAL Coursework ---
        # total_earned_points = np.sum(scores)
        # total_possible_points = num_assessments * 100
        # coursework_percentage = (total_earned_points / total_possible_points) * 100
        #
        # # The final exam grade is correlated with coursework performance, plus some randomness.
        # final_grade = coursework_percentage + np.random.normal(0, 10)
        # final_grade = np.clip(final_grade, 0, 100).round(2)
        #
        # # --- 3. Generate Repeat Information (Updated Logic) ---
        # had_to_repeat = False
        # passed_repeat = False
        # repeat_grade = np.nan
        #
        # # **UPDATED LOGIC HERE**
        # if final_grade < 40:
        #     # Student ALWAYS has to repeat if their grade is below 40.
        #     had_to_repeat = True
        #
        #     # 70% chance of passing the repeat exam.
        #     if np.random.rand() < 0.7:
        #         passed_repeat = True
        #         repeat_grade = np.random.uniform(40, 55)  # Pass grade for repeat is 40-55
        #     else:
        #         passed_repeat = False
        #         repeat_grade = np.random.uniform(20, 39)  # Fail grade for repeat is 20-39
        #
        row = [student_id] + list(scores)
        data.append(row)

    df = pd.DataFrame(data,
                      columns=['Student ID'] + assessment_columns)
    df["First Name"] = "John"
    df["Last Name"] = "Smith"
    df["Email"] = "example@example.com"

    return df


# --- Main Script ---

if not os.path.exists('dummy_data'):
    os.makedirs('dummy_data')

# Generate and save data for two different modules
df_cs101 = generate_full_student_data(150, 10, 'CS101')
df_cs101.to_csv('dummy_data/CS101.csv', index=False)
print("Generated dummy_data/CS101.csv")

df_cs204 = generate_full_student_data(100, 5, 'CS204')
df_cs204.to_csv('dummy_data/CS204.csv', index=False)
print("Generated dummy_data/CS204.csv")
