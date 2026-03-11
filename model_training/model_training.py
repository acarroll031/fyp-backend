import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

# Project root is one level up from this script's directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_prepare_data(file_root, progress_threshold):
    """
    Helper function to load training data a CSV file
    and split into training/testing sets using GroupShuffleSplit.
    """
    # Load the training data
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, file_root + progress_threshold + ".csv")
    )

    # Select features and target variable
    features = [
        "average_score",
        "assessments_completed",
        "performance_trend",
        "progress_in_semester",
        "max_consecutive_misses",
    ]
    target = "risk_score"
    groups = df[
        "Student ID"
    ]  # Use Student ID as groups to ensure no data leakage between train and test

    # Prepare X and y
    X = df[features]
    y = df[target]

    # Use GroupShuffleSplit to ensure the same student doesn't appear in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def train_random_forest(file_root, progress_threshold):
    """
    Trains and tunes a Random Forest Regressor as a baseline model.
    """
    print(f"\n--- Starting Random Forest Training ({progress_threshold}) ---")

    # Load and prepare data using the helper function
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        file_root, progress_threshold
    )

    # Hyperparameter grid for Tuning
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 20, 30, None],
        "max_features": ["sqrt", "log2", 1.0],
        "min_samples_split": [2, 5, 10],
    }

    model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1
    )

    print("Hyperparameter tuning (RF)...")
    grid_search.fit(X_train, y_train)
    print("Tuning complete!")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print(f"RF - Best Params: {grid_search.best_params_}")
    print(f"RF - Final RMSE: {rmse:.2f}")
    print(f"RF - Final R^2: {r2:.4f}")

    # Save Results
    results_df = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")
    results_df.to_csv(
        os.path.join(
            PROJECT_ROOT,
            "model_training",
            f"grid_search_results_RF_{progress_threshold}.csv",
        ),
        index=False,
    )

    # Save the best model
    model_name = f"student_risk_model_RF_{progress_threshold}.joblib"
    joblib.dump(best_model, os.path.join(PROJECT_ROOT, model_name))
    print(f"Saved RF model to {model_name}")

    return r2  # Return score for comparison


def train_xgboost(file_root, progress_threshold):
    """
    Trains and tunes an XGBoost Regressor with a refined hyperparameter grid.
    """
    print(f"\n--- Starting XGBoost Training ({progress_threshold}) ---")

    # Load and prepare data using the helper function
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        file_root, progress_threshold
    )

    # Refined hyperparameter grid based on previous results and intuition
    param_grid = {
        "n_estimators": [300, 400],
        "learning_rate": [0.01],
        "max_depth": [4, 5, 6],
        "min_child_weight": [5, 7, 9],
        "gamma": [0.1, 0.2],
        "subsample": [0.8],
        "colsample_bytree": [1.0],
    }

    # Use the same objective and random state for consistency
    model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )

    print("Hyperparameter tuning (XGBoost - Refined)...")
    grid_search.fit(X_train, y_train)
    print("Tuning complete!")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print(f"XGB - Best Params: {grid_search.best_params_}")
    print(f"XGB - Final RMSE: {rmse:.2f}")
    print(f"XGB - Final R^2: {r2:.4f}")

    # Analyze feature importances
    print("\nFeature Importances:")
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False))

    # Save Results
    results_df = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")
    results_df.to_csv(
        os.path.join(
            PROJECT_ROOT,
            "model_training",
            f"grid_search_results_XGB_{progress_threshold}.csv",
        ),
        index=False,
    )

    # Save the best model
    model_name = f"student_risk_model_{progress_threshold}.joblib"
    joblib.dump(best_model, os.path.join(PROJECT_ROOT, model_name))
    print(f"Saved refined XGBoost model to {model_name}")

    return r2  # Return score for comparison


def train_knn_model(file_root, progress_threshold):
    """
    Trains and tunes a KNeighborsRegressor model using a pipeline.
    """
    print(f"\n--- Starting KNN Training ({progress_threshold}) ---")

    # Load and prepare data using the helper function
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        file_root, progress_threshold
    )

    # Create a pipeline that includes scaling and the KNN regressor
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())])

    # Hyperparameter grid for KNN tuning
    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 11, 15],
        "knn__weights": ["uniform", "distance"],
    }

    # Use GridSearchCV to find the best hyperparameters for KNN
    grid_search = GridSearchCV(
        estimator=pipe, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )

    print("Hyperparameter tuning (KNN)...")
    grid_search.fit(X_train, y_train)
    print("Tuning complete!")

    # Evaluate the best model on the test set
    best_knn_model = grid_search.best_estimator_
    y_pred = best_knn_model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print(f"KNN - Final RMSE: {rmse:.2f}")
    print(f"KNN - Final R^2: {r2:.4f}")

    return r2  # Return score for comparison


if __name__ == "__main__":
    file = os.path.join("training_data", "Student_Data_training_")
    threshold = "0.1-1.0"  # Use the combined training data for all progress thresholds

    print("========================================")
    print("      COMPARING MODEL PERFORMANCE       ")
    print("========================================")

    # 1. Train Random Forest
    rf_score = train_random_forest(file, threshold)

    # 2. Train XGBoost
    xgb_score = train_xgboost(file, threshold)

    # 3. Train KNN
    knn_score = train_knn_model(file, threshold)

    print("\n========================================")
    print("           FINAL COMPARISON             ")
    print("========================================")
    print(f"Random Forest R^2: {rf_score:.4f}")
    print(f"XGBoost R^2:       {xgb_score:.4f}")
    print(f"KNN R^2:           {knn_score:.4f}")

    # Calculate percentage improvement of XGBoost and KNN over Random Forest
    improvement_xgb = ((xgb_score - rf_score) / rf_score) * 100
    improvement_knn = ((knn_score - rf_score) / rf_score) * 100

    # Print the improvements of XGBoost compared to Random Forest
    print("XGBoost vs Random Forest:")
    if improvement_xgb > 0:
        print(f"XGBoost improved performance by {improvement_xgb:.2f}%")
    else:
        print(f"Random Forest performed better by {abs(improvement_xgb):.2f}%")
    print("========================================")
    # Print the improvements of KNN compared to Random Forest
    print("KNN vs Random Forest:")
    if improvement_knn > 0:
        print(f"KNN improved performance by {improvement_knn:.2f}%")
    else:
        print(f"Random Forrest performed better by {abs(improvement_knn):.2f}%")
    print("========================================")
