# src/app.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    """
    Main function to load data, train a Random Forest model with optimized
    hyperparameters, evaluate it, and save the trained model.
    """
    print("Starting Random Forest Diabetes Prediction Application...\n")

    # --- Configuration ---
    # Set a random seed for reproducibility across all steps
    RANDOM_STATE = 42
    # Define the proportion of the dataset to be used as test set
    TEST_SIZE = 0.2
    # Raw URL for the processed diabetes dataset on GitHub
    FILE_PATH = 'https://raw.githubusercontent.com/RozaSekouri/Random-Forest-Roza/main/data/processed/diabetes_processed.csv'
    # Directory to save the trained model
    MODELS_DIR = 'models'
    # Filename for the saved model
    MODEL_FILENAME = os.path.join(MODELS_DIR, 'random_forest_diabetes_model.pkl')

    # --- Step 1: Loading the dataset ---
    print("--- Step 1: Loading the dataset ---")
    try:
        # Load the processed dataset directly from the GitHub raw URL
        df_processed = pd.read_csv(FILE_PATH)

        # Assuming 'Outcome' is the target variable (0 for non-diabetic, 1 for diabetic)
        # All other columns are features
        X = df_processed.drop('Outcome', axis=1)
        y = df_processed['Outcome']

        # Split the dataset into training and testing sets
        # stratify=y ensures that the proportion of target classes is the same
        # in both training and testing sets as in the original dataset.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print("Dataset loaded and split successfully from GitHub URL.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    except Exception as e:
        print(f"Error loading dataset from URL: {e}")
        print("Please ensure the URL is correct and the file is accessible.")
        # Exit the application if data loading fails, as subsequent steps depend on it
        exit()

    # Display basic information about the loaded data for verification
    print("\nX_train head:")
    print(X_train.head())
    print("\ny_train value counts:")
    print(y_train.value_counts(normalize=True))

    # --- Step 2: Build and train the Random Forest model ---
    print("\n--- Step 2: Building and Training the Random Forest Model ---")

    # Define the best hyperparameters found during the exploration phase (from GridSearchCV results)
    # These values are taken directly from your previous successful run's output:
    # Best parameters found by GridSearchCV: {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    best_n_estimators = 200
    best_max_depth = 15
    best_min_samples_split = 2
    best_min_samples_leaf = 1

    print(f"\nTraining Random Forest with the following optimized parameters:")
    print(f"  n_estimators={best_n_estimators}")
    print(f"  max_depth={best_max_depth}")
    print(f"  min_samples_split={best_min_samples_split}")
    print(f"  min_samples_leaf={best_min_samples_leaf}")

    # Initialize the RandomForestClassifier with the best parameters
    # n_jobs=-1 utilizes all available CPU cores for faster training
    rf_final_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        random_state=RANDOM_STATE, # Ensure reproducibility
        n_jobs=-1
    )

    # Train the model on the training data
    rf_final_model.fit(X_train, y_train)
    print("\nRandom Forest model training complete.")

    # --- Evaluate the trained model on the test set ---
    print("\n--- Model Evaluation on Test Set ---")
    y_pred_final = rf_final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)

    print(f"Final Random Forest Accuracy on Test Set: {final_accuracy:.4f}")
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred_final))
    print("\nFinal Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))

    # --- Step 3: Save the model ---
    print("\n--- Step 3: Saving the Model ---")
    # Create the 'models' directory if it does not already exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        # Save the trained model using pickle
        with open(MODEL_FILENAME, 'wb') as file:
            pickle.dump(rf_final_model, file)
        print(f"Model successfully saved to: {MODEL_FILENAME}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nRandom Forest Diabetes Prediction Application Completed.")

# This ensures that main() is called only when the script is executed directly
if __name__ == "__main__":
    main()