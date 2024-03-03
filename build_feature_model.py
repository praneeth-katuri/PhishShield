import pickle
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from preprocessing.url_feature_extraction import FeatureExtractor
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Function to load data


def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['Index'], axis=1)
    X = data.drop(["class"], axis=1)
    y = data["class"]
    return X, y

# Function to split data


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Define parameter grids for GridSearchCV
param_grids = {
    "Logistic Regression": {'solver': ['liblinear', 'lbfgs'], 'C': [0.01, 0.1, 1.0]},
    "Support Vector Machine": {'C': [0.001, 0.01, 0.1, 1, 10.0], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']},
    "Random Forest": {'n_estimators': [10, 50, 100, 200], 'max_depth': [3, 5, 7, 10]},
    "Gradient Boosting": {'learning_rate': [0.01, 0.001, 0.1], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
}

# Function to train and evaluate a model


def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    f1_train = metrics.f1_score(y_train, y_train_pred)
    f1_test = metrics.f1_score(y_test, y_test_pred)
    recall_train = metrics.recall_score(y_train, y_train_pred)
    recall_test = metrics.recall_score(y_test, y_test_pred)
    precision_train = metrics.precision_score(y_train, y_train_pred)
    precision_test = metrics.precision_score(y_test, y_test_pred)
    return acc_train, acc_test, f1_train, f1_test, recall_train, recall_test, precision_train, precision_test

# Function to store model results


def store_results(ML_Model, accuracy_train, accuracy_test, f1_score_train, f1_score_test, recall_train, recall_test, precision_train, precision_test):
    results = {"ML Model": ML_Model, "Accuracy (Train)": accuracy_train, "Accuracy (Test)": accuracy_test, "F1 Score (Train)": f1_score_train, "F1 Score (Test)": f1_score_test, "Recall (Train)": recall_train, "Recall (Test)": recall_test, "Precision (Train)": precision_train, "Precision (Test)": precision_test}
    return results

# Function to perform upsampling of minority class


def upsample_data(X_train, y_train):
    # Initialize SMOTE
    smote = SMOTE(random_state=42)

    # Resample the minority class
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled

# Main function


def main():
    # Load data
    X, y = load_data("datafiles/dataset_for_feature_model.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Upsample minority class
    X_train, y_train = upsample_data(X_train, y_train)

    # Initialize models with GridSearchCV
    models = {
        "Logistic Regression": GridSearchCV(LogisticRegression(), param_grids["Logistic Regression"], cv=5),
        "Support Vector Machine": GridSearchCV(SVC(), param_grids["Support Vector Machine"], cv=5),
        "Random Forest": GridSearchCV(RandomForestClassifier(), param_grids["Random Forest"], cv=5, n_jobs=-1),
        "Gradient Boosting": GridSearchCV(GradientBoostingClassifier(), param_grids["Gradient Boosting"], cv=5)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # Train and evaluate models
    results = []
    for name, model in models.items():
        acc_train, acc_test, f1_train, f1_test, recall_train, recall_test, precision_train, precision_test = train_evaluate_model(model, X_train, X_test, y_train, y_test)
        if acc_test > best_accuracy:
            best_accuracy = acc_test
            best_model = model
            best_model_name = name
        results.append(store_results(name, acc_train, acc_test, f1_train, f1_test, recall_train, recall_test, precision_train, precision_test))

        # Generate classification report
        y_test_pred = model.predict(X_test)
        print(f"Classification Report for {name}:")
        print(metrics.classification_report(y_test, y_test_pred, target_names=('Phishing', 'Legitimate')))

    # Save the best performing model
    if best_model is not None:
        ml_pipeline = Pipeline([
            ('feature_extraction', FeatureExtractor()),  # Feature extraction step
            ('model', best_model)  # Machine learning model step
        ])
        with open('models/feature_model.pkl', 'wb') as f:
            pickle.dump(ml_pipeline, f)
        print("Best performing model saved as 'feature_model.pkl'.")
        print(f"The best performing model is: {best_model_name}, with accuracy: {best_accuracy}")

    # Display results
    results_df = pd.DataFrame(results)
    sorted_results = results_df.sort_values(by=['Accuracy (Test)', 'F1 Score (Test)'], ascending=False).reset_index(drop=True)
    print(sorted_results)


if __name__ == "__main__":
    main()
