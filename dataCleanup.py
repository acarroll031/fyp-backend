import pandas as pd

# Load the CSV file into a DataFrame
studentLabs = pd.read_csv("studentsLabs.csv")

# Remove identifying columns
studentLabs = studentLabs.drop(["First Name", "Last Name", "Email"], axis = 1)

# Convert lab scores from 0-4 scale to 0-100 scale
studentLabs.loc[:, 'Lab 1':'Lab 10'] = studentLabs.loc[:, 'Lab 1':'Lab 10'] * 25

# Define lab columns
lab_columns = ["Lab 1", "Lab 2", "Lab 3", "Lab 4", "Lab 5", "Lab 6", "Lab 7", "Lab 8", "Lab 9", "Lab 10"]

# Calculate the average grade across all labs
studentLabs["averageGrade"] = studentLabs[lab_columns].mean(axis=1)

# Count the number of labs completed (score > 0)
studentLabs["labsCompleted"] = studentLabs[lab_columns].gt(0).sum(axis=1)

# Determine performance trend: average of first half vs second half
first_half = studentLabs[lab_columns[:5]].mean(axis=1)
second_half = studentLabs[lab_columns[5:]].mean(axis=1)
studentLabs["performanceTrend"] = second_half - first_half

# Round all numerical values to 2 decimal places and save to a new CSV file
studentLabs = studentLabs.round(2)
studentLabs.to_csv("studentLabsTransformed.csv", index = False)

print(studentLabs.head())