import pandas as pd
import numpy as np

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

    # Remove identifying columns
    student_labs = student_labs.drop(["First Name", "Last Name", "Email"], axis=1)

    # Reshape the DataFrame from wide to long format
    student_labs = pd.melt(student_labs, id_vars=["Student ID"], var_name="assessment_name", value_name="score")

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
# preprocess_module_data("dummy_data/CS101.csv", "CS101", total_assessments=10, assessment_total_score=4)

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

    student_labs = student_labs[student_labs["progress_in_semester"] <= progress_threshold]

    student_labs = student_labs.groupby("Student ID").agg(average_score=("score", "mean"),
                                                          assessments_completed=("score", lambda x: (x >0).sum()),
                                                            performance_trend=("score", calculate_performance_trend)
                                                          )
    student_labs = student_labs.round(2)
    # Create a new filename for the training data
    base_filename = csv_file.split(".csv")[0]
    new_filename = base_filename + "_training.csv"
    student_labs.to_csv(new_filename, index=False)

    print(student_labs)

training_data(csv_file="studentsLabs_normalised.csv", progress_threshold=1)
