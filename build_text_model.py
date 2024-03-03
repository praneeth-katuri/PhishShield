import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing.preprocess_url import URLPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import pickle
from imblearn.over_sampling import SMOTE

# Load the dataset


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    X = df['url']
    y = df['status']
    return X, y

# Split the dataset into training and testing sets


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Define parameter grids for GridSearchCV
param_grids = {
    "Logistic Regression": {'C': [0.1, 1.0, 10.0]},
    "Support Vector Machine": {'C': [0.1, 1.0, 10.0], 'gamma': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']},
    "Random Forest": {'n_estimators': [100, 200, 300]},
    "Gradient Boosting": {'n_estimators': [100, 200, 300]}
}

# Function to train and evaluate a model


def train_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    f1_train = metrics.f1_score(y_train, y_train_pred)
    f1_test = metrics.f1_score(y_test, y_test_pred)
    recall_train = metrics.recall_score(y_train, y_train_pred)
    recall_test = metrics.recall_score(y_test, y_test_pred)
    precision_train = metrics.precision_score(y_train, y_train_pred)
    precision_test = metrics.precision_score(y_test, y_test_pred)
    return acc_train, acc_test, f1_train, f1_test, recall_train, recall_test, precision_train, precision_test


def store_results(ML_Model, accuracy_train, accuracy_test, f1_score_train, f1_score_test, recall_train, recall_test, precision_train, precision_test):
    results = {"ML Model": ML_Model, "Accuracy (Train)": accuracy_train, "Accuracy (Test)": accuracy_test, "F1 Score (Train)": f1_score_train, "F1 Score (Test)": f1_score_test,
               "Recall (Train)": recall_train, "Recall (Test)": recall_test, "Precision (Train)": precision_train, "Precision (Test)": precision_test}
    return results

# Save the best model to a file


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Function to perform upsampling of minority class


def upsample_data(X_train, y_train):
    # Initialize SMOTE
    smote = SMOTE(random_state=42)

    # Resample the minority class
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled


def main():
    # Load data
    X, y = load_data('datafiles/dataset_for_text_model.csv')

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Upsample minority class
    X_train, y_train = upsample_data(X_train, y_train)

    # Define pipeline for each classifier
    classifiers = {
        "Logistic Regression": Pipeline([
            ('preprocessor', URLPreprocessor()),
            ('vectorizer', CountVectorizer(tokenizer=None,
             stop_words=None, lowercase=False, ngram_range=(1, 2))),
            ('classifier', GridSearchCV(LogisticRegression(),
             param_grids["Logistic Regression"], cv=5, scoring='accuracy'))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', URLPreprocessor()),
            ('vectorizer', CountVectorizer(tokenizer=None,
             stop_words=None, lowercase=False, ngram_range=(1, 2))),
            ('classifier', GridSearchCV(RandomForestClassifier(),
             param_grids["Random Forest"], cv=5, scoring='accuracy'))
        ]),
        "Gradient Boosting": Pipeline([
            ('preprocessor', URLPreprocessor()),
            ('vectorizer', CountVectorizer(tokenizer=None,
             stop_words=None, lowercase=False, ngram_range=(1, 2))),
            ('classifier', GridSearchCV(GradientBoostingClassifier(),
             param_grids["Gradient Boosting"], cv=5, scoring='accuracy'))
        ])
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # Train and evaluate models
    results = []
    for name, model in classifiers.items():
        acc_train, acc_test, f1_train, f1_test, recall_train, recall_test, precision_train, precision_test = train_evaluate_model(
            model, X_train, X_test, y_train, y_test)
        if acc_test > best_accuracy:
            best_accuracy = acc_test
            best_model = model
            best_model_name = name
        results.append(store_results(name, acc_train, acc_test, f1_train,
                       f1_test, recall_train, recall_test, precision_train, precision_test))

        # Generate classification report
        y_test_pred = model.predict(X_test)
        print(f"Classification Report for {name}:")
        print(metrics.classification_report(y_test, y_test_pred,
                                            target_names=('Phishing', 'Legitimate')))
    # Save the best model
    if best_model is not None:
        save_model(best_model, 'text_model.pkl')
        print(f"Best performing model saved as 'text_model.pkl'. Classifier: {
              best_model_name}, Accuracy: {best_accuracy}")

    # Display results
    results_df = pd.DataFrame(results)
    sorted_results = results_df.sort_values(
        by=['Accuracy (Test)', 'F1 Score (Test)'], ascending=False).reset_index(drop=True)
    print(sorted_results)


if __name__ == "__main__":
    main()
