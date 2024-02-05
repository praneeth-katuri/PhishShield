import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(["class"], axis=1)
    y = data["class"]
    return X, y

def upsample_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    X, y = load_data("datafiles/dataset_for_feature_model.csv")
    X_train, y_train = upsample_data(X_train, y_train)


    param_grids = {
    "Logistic Regression": {'C': [1, 10]},
    "Support Vector Machine": {'C': [1, 10]},
    "Random Forest": {'n_estimators': [100, 200]}
}

    models = {
    "Logistic Regression": GridSearchCV(LogisticRegression(), param_grids["Logistic Regression"], cv=3),
    "Support Vector Machine": GridSearchCV(SVC(), param_grids["Support Vector Machine"], cv=3),
    "Random Forest": GridSearchCV(RandomForestClassifier(), param_grids["Random Forest"], cv=3)
}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Results for {name}:")
        print(metrics.classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
