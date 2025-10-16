import pandas as pd 
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# Config
DATA_PATH = 'data/dataset.csv'
MODEL_PATH = 'models/random_forest_model.joblib'
METRICS_PATH = 'models/metrics.json'
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)


def train_and_evaluate():
    print("Starting training and evaluation process...")
    # Load data
    data = pd.read_csv(DATA_PATH)
    print(f"Data shape: {data.shape}")
    print("Data load successful.")


    # Preprocess data
    print("Starting data preprocessing...")
    X = data.drop('Drug', axis=1)
    y = data['Drug']

    encoder = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    transformer = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_features),
            ('cat', encoder, categorical_features)
        ])

    print("Data preprocessing completed.")

    # Split data
    print("Splitting data into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # Create model
    model = Pipeline(steps=[
        ('preprocessor', transformer),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    # Define parameters for GridSearchCV
    print("Starting model training with hyperparameter tuning...")
    grid_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model, grid_params, cv=5, n_jobs=-1, verbose=2)

    # Fit model
    print("Fitting the model...")
    grid_search.fit(X_train, y_train)
    print("Model training completed.")
    print(f"Best parameters found: {grid_search.best_params_}")

    # predict
    print("Making predictions on the test set...")
    y_pred = grid_search.predict(X_test)
    class_report_str = classification_report(y_test, y_pred)
    print("Classification Report:")

    # print classification report
    print(class_report_str)
    
    # Print confusion matrix]
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
  
    # Save results
    print("Saving model and metrics...")
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    json_metrics = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(json_metrics).to_json(METRICS_PATH)





    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

if __name__ == "__main__":
    train_and_evaluate()
    print("Training and evaluation process completed.")