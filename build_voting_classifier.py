import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

data0 = pd.read_csv('datfiles/dataset_for_voting_classifier.csv')

# Load the first model
text_model = joblib.load('models/text_model.pkl')

# Load the second model
feature_model = joblib.load('models/feature_model.pkl')

# Create a voting ensemble classifier
voting_classifier = VotingClassifier(estimators=[
    ('pipeline1', pipeline1),
    ('pipeline2', pipeline2)
], voting='hard')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the voting classifier to your training data
voting_classifier.fit(X_train, y_train)

# Make predictions using the voting classifier
predictions = voting_classifier.predict(X_test)

# Evaluate performance
accuracy = voting_classifier.score(X_test, y_test)
print("Accuracy:", accuracy)
