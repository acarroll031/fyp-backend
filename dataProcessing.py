import hashlib

import pandas as pd
import numpy as np
from defusedxml.lxml import tostring

def anonymise_id(student_id, salt):
    # Combine student_id and salt, encode, and hash
    return hashlib.sha256(f"{student_id}_{salt}".encode()).hexdigest()[:16]

def merge_csv_files(file1, file2, output_file):
    """
    Merge two CSV files into one DataFrame and save to a new CSV file.
    :param file1: str, path to the first CSV file
    :param file2: str, path to the second CSV file
    :param output_file: str, path to the output CSV file
    """
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Merge the DataFrames on common columns (if any)
    merged_df = pd.merge(df1, df2, how='outer')

    # Drop rows with any missing values
    merged_df = merged_df.dropna()

    # Generate random Student IDs (8-digit numbers)
    salt = output_file  # Use a different salt for each dataset
    merged_df['Student ID'] = merged_df['Student ID'].apply(lambda x: anonymise_id(x, salt))

    # Round scores to 2 decimal places
    merged_df.round(2)

    # merged_df = merged_df.drop(["Email", "Total"], axis=1)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")

#merge_csv_files("CS161_Data/CS161_Exam_Totals_2.csv", "CS161_Data/CS161_Lab_Totals_2.csv", "CS161_Data/CS161_Combined_Totals_2.csv")

def preprocess_module_data(csv_file, module_code, total_assessments, assessment_total_score):
    """
    Preprocess the student lab data for a given module.
    This includes removing identifying information, reshaping the data,
    normalising scores, and adding progress information.
    The cleaned data is saved to a new CSV file.
    :param csv_file: str, path to the input CSV file
    :param module_code: str, the module code to add to the data
    :param total_assessments: int, total number of assessments in the module
    :param assessment_total_score: int, the total score possible for each assessment
    """
    # Load the CSV file into a DataFrame
    student_labs = pd.read_csv(csv_file)

    # Reshape the DataFrame from wide to long format
    student_labs = pd.melt(student_labs, id_vars=["Student ID", "TOTAL"], var_name="assessment_name", value_name="score")

    # Add a new column for module code
    student_labs.insert(1, "module_code", module_code)

    # Extract lab number from assessment_name and rename the column
    student_labs["assessment_name"] = student_labs["assessment_name"].str.extract(r'(\d+)').astype(int)
    student_labs = student_labs.rename(columns={"assessment_name": "assessment_number"})

    # Add a new column for progress in semester
    student_labs["progress_in_semester"] = student_labs["assessment_number"] / total_assessments

    # Normalise scores to be out of 100
    student_labs["score"] = student_labs["score"] / assessment_total_score * 100

    # Round scores to 2 decimal places
    student_labs.round(2)

    # Create a new filename for the cleaned data
    base_filename = csv_file.split(".csv")[0]
    new_filename = base_filename + "_normalised.csv"

    # Save the cleaned DataFrame to a new CSV file
    student_labs.to_csv(new_filename, index=False)

    print(student_labs.head())

# Example usage
#preprocess_module_data("CS161_Data/CS161_Combined_Totals_2.csv", "CS161", total_assessments=10, assessment_total_score=4)


def calculate_performance_trend(student_scores):
    """
    Calculate the performance trend of a student based on their scores.
    The trend is determined by comparing the average of the first half of the scores
    to the average of the second half. A positive trend indicates improvement, while
    a negative trend indicates decline.
    :param student_scores:
    :return float: Trend value (positive for improvement, negative for decline, zero for no change)
    """
    # Filter out non-submitted assignments for a more accurate trend
    scores = [score for score in student_scores if score > 0]

    num_scores = len(scores)

    # A trend cannot be calculated with fewer than 2 data points
    if num_scores < 2:
        return 0

    # Split the scores into two halves
    mid_point = num_scores // 2
    first_half = scores[:mid_point]
    second_half = scores[-mid_point:]

    # Calculate the average of each half
    first_avg = np.mean(first_half) if first_half else 0
    second_avg = np.mean(second_half) if second_half else 0

    return second_avg - first_avg

def calculate_risk_score(final_grade):
    """
    Calculate a risk score based on the final grade.
    The risk score is higher for lower final grades, with an additional penalty
    for grades below 40.
    :param final_grade: Series, the final grade of the student
    :return float: risk score (0 to 100+)
    """
    final_grade = final_grade.iloc[0]
    risk_score = 100-final_grade

    # if final_grade < 40:
    #     risk_score += 40

    return risk_score

def calculate_max_consecutive_misses(student_scores):
    """
    Calculate the maximum number of consecutive missed assessments (score of 0).
    :param student_scores: Series, the scores of the student
    :return int: maximum number of consecutive misses
    """
    max_misses = 0
    current_misses = 0

    for score in student_scores:
        if score == 0:
            current_misses += 1
            max_misses = max(max_misses, current_misses)
        else:
            current_misses = 0

    return max_misses



def training_data(csv_file, progress_threshold):
    """
    Generate training data by filtering student lab data based on progress threshold
    and calculating average scores, assessments completed, and performance trends.
    ******************************************************************************
    : This function reads a CSV file containing student lab data, filters the data
    : to include only those assessments completed up to a specified progress threshold
    : in the semester, and then computes key metrics for each student. These metrics
    : include the average score across completed assessments, the number of assessments
    : completed, and the trend in performance over time. The performance trend is
    : calculated by comparing the average scores of the first half of the completed
    : assessments to the second half, providing insight into whether a student's
    : performance is improving, declining, or remaining stable.
    ******************************************************************************
    :param csv_file: str, path to the input CSV file
    :param progress_threshold: float, threshold for progress in semester (0 to 1)
    """

    student_labs = pd.read_csv(csv_file)

    # Filter the DataFrame based on the progress threshold
    student_labs = student_labs[student_labs["progress_in_semester"] <= progress_threshold]

    # Group by Student ID and calculate required metrics
    student_labs = student_labs.groupby("Student ID").agg(average_score=("score", "mean"),
                                                          assessments_completed=("score", lambda x: (x > 0).sum()),
                                                          performance_trend=("score", calculate_performance_trend),
                                                          risk_score=("TOTAL", calculate_risk_score),
                                                          progress_in_semester=("progress_in_semester", "max"),
                                                          max_consecutive_misses=("score", calculate_max_consecutive_misses)
                                                          ).reset_index()


    # Round scores to 2 decimal places
    student_labs = student_labs.round(2)

    # Create a new filename for the training data
    base_filename = csv_file.split(".csv")[0]
    new_filename = "trainingData/" + base_filename + "_training_" + str(progress_threshold) + ".csv"
    student_labs.to_csv(new_filename, index=False)

    print(student_labs)

# Generate training data for progress thresholds from 0.1 to 1.0
# for i in range(1, 11):
#     progress_threshold = i / 10
#     training_data(csv_file="CS161_Data/CS161_Combined_Totals_1_normalised.csv", progress_threshold=progress_threshold)

def combine_training_data(file_list, output_file):
    """
    Combine multiple training data CSV files into one DataFrame and save to a new CSV file.
    :param file_list: list of str, paths to the input CSV files
    :param output_file: str, path to the output CSV file
    """
    # Load and concatenate the CSV files into a single DataFrame
    df_list = [pd.read_csv(file) for file in file_list]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined training data saved to {output_file}")


def convert_grades_to_students(grades_df):
    """Convert grades data to students schema format."""

    # Group by student and calculate metrics
    students_df = grades_df.groupby(["student_id", 'module']).agg(
        student_name=("student_name", "first"),
        average_score=("score", "mean"),
        assessments_completed=("assessment_number", "count"),
        performance_trend=("score", calculate_performance_trend),
        progress_in_semester=("progress_in_semester", "max"),
        max_consecutive_misses=("score", calculate_max_consecutive_misses)
    ).reset_index()

    # Round average grade to 2 decimal places
    students_df = students_df.round(2)

    return students_df[[
        'student_id',
        'student_name',
        'module',
        'average_score',
        'assessments_completed',
        'performance_trend',
        'max_consecutive_misses',
        'progress_in_semester'
    ]]


# Example usage
#file_list = [f"trainingData/CS161_Data/CS161_Combined_Totals_1_normalised_training_{i/10}.csv" for i in range(1, 11)]
#combine_training_data(file_list, "trainingData/CS161_Data/CS161_Combined_Totals_1_normalised_training_0.1-1.0.csv")