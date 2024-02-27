import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
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

# Main function
def main():
    # Load data
    X, y = load_data("datafiles/phishing.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": GridSearchCV(SVC(), {'gamma': [0.1], 'kernel': ['rbf', 'linear']}),
        "Naive Bayes": GaussianNB(),# low performance
        #"Decision Tree": DecisionTreeClassifier(),#redudency
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        #"CatBoost": CatBoostClassifier(iterations=100, verbose=False),
        #"MLP": MLPClassifier()
    }

    best_model = None
    best_accuracy = 0
    best_model_name= ""

    # Train and evaluate models
    results = []
    for name, model in models.items():
        acc_train, acc_test, f1_train, f1_test, recall_train, recall_test, precision_train, precision_test = train_evaluate_model(model, X_train, X_test, y_train, y_test)
        if acc_test > best_accuracy:
            best_accuracy = acc_test
            best_model = model
            best_model_name = name
        results.append(store_results(name, acc_train, acc_test, f1_train, f1_test, recall_train, recall_test, precision_train, precision_test))

    # Save the best performing model
    if best_model is not None:
        with open('models/feature_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print("Best performing model saved as 'feature_model.pkl'.")
        print(f"The best performing model is: {best_model_name}, with accuracy: {best_accuracy}")

    # Display results
    results_df = pd.DataFrame(results)
    sorted_results = results_df.sort_values(by=['Accuracy (Test)', 'F1 Score (Test)'], ascending=False).reset_index(drop=True)
    print(sorted_results)

        # Print classification report for each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        print(f"Classification Report for {name}:")
        print(metrics.classification_report(y_test, y_test_pred,target_names=('Phishing','Legitimate')))

if __name__ == "__main__":
    main()
