import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from dataProcessing import progress_threshold


def train_model(file_root, progress_threshold):
    """
    Train a RandomForestRegressor model to predict student risk scores based on
    features such as average score, assessments completed, performance trend,
    and progress in semester. The trained model is saved to a file for future use.
    """
    # Load the training data
    df = pd.read_csv(file_root + progress_threshold + ".csv")

    # Select features and target variable
    features = ["average_score", "assessments_completed", "performance_trend", "progress_in_semester"]
    target = "risk_score"

    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [10, 50, 100, 200, 300, 400, 500],  # Number of trees in the forest
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, None],  # Maximum depth of the trees (None means no limit)
    }

    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

    print("Starting hyperparameter tuning with GridSearchCV... (This may take a while)")
    grid_search.fit(X_train, y_train)

    print("\nTuning complete!")

    results = grid_search.cv_results_
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='rank_test_score')
    results_df.to_csv(file_root + "grid_search_results_" + progress_threshold + ".csv", index=False)

    # Print the best combination of parameters it found
    print(f"Best parameters found: {grid_search.best_params_}")

    # Print the best R-squared score it achieved during cross-validation
    print(f"Best R-squared score from Grid Search: {grid_search.best_score_:.2f}")

    # The grid_search object itself is now the best model, already trained on all the training data.
    best_model = grid_search.best_estimator_

    # You can now save this best_model to a file
    joblib.dump(best_model, "best_student_risk_model.joblib")

    # # Make predictions
    # y_pred = model.predict(X_test)
    # rmse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    #
    # print(f"Root Mean Squared Error: {rmse}")
    # print(f"R^2 Score: {r2}")

    # Save the trained model
    # model_name = "student_risk_model_" + progress_threshold + ".pkl"
    # joblib.dump(model, model_name)

# for i in range(1, 11):
#     progress_threshold = str(i / 10)
#     train_model(file_root="trainingData/CS161_Data/CS161_Combined_Totals_2_normalised_training_", progress_threshold=progress_threshold)

train_model(file_root="trainingData/CS161_Data/CS161_Combined_Totals_2_normalised_training_", progress_threshold="0.1-1.0")