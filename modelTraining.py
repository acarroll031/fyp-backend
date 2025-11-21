import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

def train_model(file_root_1,file_root_2, progress_threshold):
    """
    Train a RandomForestRegressor model to predict student risk scores based on
    features such as average score, assessments completed, performance trend,
    and progress in semester. The trained model is saved to a file for future use.
    """
    # Load the training data
    df = pd.read_csv(file_root_1 + progress_threshold + ".csv")
    df2 = pd.read_csv(file_root_2 + progress_threshold + ".csv")
    df_combined = pd.concat([df, df2], ignore_index=True)

    print(df_combined)

    # Select features and target variable
    features = ["average_score", "assessments_completed", "performance_trend", "progress_in_semester", "max_consecutive_misses"]
    target = "risk_score"
    groups = df_combined["Student ID"]

    X = df_combined[features]
    y = df_combined[target]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'max_features': ['sqrt', 'log2', 1.0],  # Test different feature selection strategies
        'min_samples_split': [2, 5, 10]  # Control node splitting to prevent overfitting
    }

    # param_grid = {
    #     'n_estimators': [100, 300, 500],
    #     'max_depth': [10, 20, 30, None],
    # }


    # Initialize and train the model

    model = RandomForestRegressor(random_state=42)
    #  model = XGBRegressor(random_state=42)

    #model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

    print("Starting hyperparameter tuning with GridSearchCV... (This may take a while)")
    grid_search.fit(X_train, y_train)
    print("\nTuning complete!")

    # --- Save and Print Tuning Results ---
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best R-squared score from cross-validation: {grid_search.best_score_:.4f}")

    # Convert results to a DataFrame and save
    results_df = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')
    results_df.to_csv(f"grid_search_results_{progress_threshold}.csv", index=False)
    print(f"Tuning results saved to grid_search_results_{progress_threshold}.csv")

    # Print the best combination of parameters it found
    print(f"Best parameters found: {grid_search.best_params_}")

    # Print the best R-squared score it achieved during cross-validation
    print(f"Best R-squared score from Grid Search: {grid_search.best_score_:.2f}")

    print("\n--- Final Model Evaluation on the Unseen Test Set ---")
    best_model = grid_search.best_estimator_

    # Make predictions on the test data
    y_pred = best_model.predict(X_test)

    # Calculate final error metrics
    mse = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(mse)
    final_r2 = r2_score(y_test, y_pred)

    print(f"Final Root Mean Squared Error (RMSE): {final_rmse:.2f}")
    print(f"Final R^2 Score: {final_r2:.4f}")

    # Save the final, best-performing model
    model_name = f"student_risk_model_{progress_threshold}.joblib"
    joblib.dump(best_model, model_name)
    print(f"Final best model saved to {model_name}")


def train_knn_model(file_root_1, file_root_2, progress_threshold):
    """
    Trains and tunes a KNeighborsRegressor model using a pipeline
    to properly scale the features first.
    """
    # ... (Your data loading and combining code is the same) ...
    df1 = pd.read_csv(file_root_1 + progress_threshold + ".csv")
    df2 = pd.read_csv(file_root_2 + progress_threshold + ".csv")
    df_combined = pd.concat([df1, df2], ignore_index=True)

    features = ["average_score", "assessments_completed", "performance_trend", "progress_in_semester", "max_consecutive_misses"]
    target = "risk_score"
    groups = df_combined["Student ID"]
    X = df_combined[features]
    y = df_combined[target]

    # ... (Your GroupShuffleSplit is the same) ...
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- 1. Create the Pipeline ---
    # This pipeline defines the steps: first, scale the data,
    # then, run the KNN model.
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])

    # --- 2. Define the Hyperparameter Grid for the Pipeline ---
    # When tuning a pipeline, you must prefix the parameter names
    # with the step name (e.g., 'knn__n_neighbors').
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 11, 15],  # The 'K' value (how many neighbors)
        'knn__weights': ['uniform', 'distance']  # 'distance' gives more weight to closer neighbors
    }

    # --- 3. Run GridSearchCV on the Pipeline ---
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)

    print("Starting hyperparameter tuning for KNN...")
    grid_search.fit(X_train, y_train)
    print("\nTuning complete!")

    # --- 4. Evaluate the Best KNN Model ---
    print(f"Best KNN parameters found: {grid_search.best_params_}")
    print(f"Best KNN R-squared score from cross-validation: {grid_search.best_score_:.4f}")

    best_knn_model = grid_search.best_estimator_
    y_pred = best_knn_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(mse)
    final_r2 = r2_score(y_test, y_pred)

    print("\n--- Final KNN Model Evaluation on the Unseen Test Set ---")
    print(f"Final Root Mean Squared Error (RMSE): {final_rmse:.2f}")
    print(f"Final R^2 Score: {final_r2:.4f}")

# for i in range(1, 11):
#     progress_threshold = str(i / 10)
#     train_model(file_root="trainingData/CS161_Data/CS161_Combined_Totals_2_normalised_training_", progress_threshold=progress_threshold)

train_knn_model(
    file_root_1="trainingData/CS161_Data/CS161_Combined_Totals_1_normalised_training_",
    file_root_2="trainingData/CS161_Data/CS161_Combined_Totals_2_normalised_training_" ,
    progress_threshold="0.1-1.0"
)

train_model(
    file_root_1="trainingData/CS161_Data/CS161_Combined_Totals_1_normalised_training_",
    file_root_2="trainingData/CS161_Data/CS161_Combined_Totals_2_normalised_training_" ,
    progress_threshold="0.1-1.0"
)