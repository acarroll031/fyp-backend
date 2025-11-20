import pandas as pd
import numpy as np


def generate_assessment_data_with_misses(source_file, num_files=12, missing_rate=0.10):
    """
    Generates assessment files where some students get 0 because they didn't turn up.
    missing_rate: The probability (0.0 to 1.0) that a student misses the exam.
                  0.10 means roughly 10% of students will get a 0.
    """
    # 1. Load students
    df_source = pd.read_csv(source_file)
    students_base = df_source[['student_id', 'student_name']].drop_duplicates()

    print(f"Generating files with a {int(missing_rate * 100)}% missing rate...")

    for i in range(1, num_files + 1):
        df_current = students_base.copy()
        df_current['assessment_number'] = i

        # 2. Generate base scores (Normal Distribution)
        scores = np.random.normal(loc=65, scale=15, size=len(df_current))
        scores = np.clip(scores, 0, 100)

        # 3. Apply "Not Turn Up" Logic
        # Create a boolean mask: True if student misses, False if they attend
        # np.random.random() generates a float between 0.0 and 1.0 for each row
        missed_mask = np.random.random(len(df_current)) < missing_rate

        # Overwrite the scores where the mask is True with 0
        scores[missed_mask] = 0

        df_current['score'] = np.round(scores, 2)

        # 4. Save
        filename = f'assessment_{i}.csv'
        df_current.to_csv(filename, index=False)

        # Optional: Print how many missed this specific assessment
        miss_count = np.sum(missed_mask)
        print(f" -> Generated: {filename} ({miss_count} students missed)")


if __name__ == "__main__":
    generate_assessment_data_with_misses('test_upload.csv')