import pandas as pd

def preprocess_module_data(csv_file, module_code, total_assessments, assessment_total_score):
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
preprocess_module_data("dummy_data/CS101.csv", "CS101", total_assessments=10, assessment_total_score=4)