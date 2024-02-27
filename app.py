import pickle
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect
import numpy as np
import pandas as pd
from feature_extraction import featureExtraction

app = Flask(__name__)  # Creating Flask application instance
csrf = CSRFProtect(app)  # Adding CSRF protection to the app

# Loading the trained models
text_model = pickle.load(open("models/text_model.pkl", "rb"))  # Loading text-based phishing detection model
feature_model = pickle.load(open("models/feature_model.pkl", "rb"))  # Loading feature-based phishing detection model

@app.route('/')  # Defining route for the home page
def home():
    """
    Renders the home page.

    Returns:
        rendered HTML template
    """
    return render_template('index.html')  # Rendering the HTML template for the home page

@app.route('/detect', methods=['POST'])  # Defining route for phishing detection
@csrf.exempt  # Exempting CSRF protection for this route
def detect_phishing():
    """
    Detects phishing URLs.

    Returns:
        rendered HTML template with the prediction result
    """
    if request.method == 'POST':
        url = request.form["url"]  # Getting URL from the form data

        features_df = featureExtraction(url)  # Extracting features from the URL using a function

        feature_pred = feature_model.predict(features_df)  # Making prediction using feature-based model
        print(feature_pred[0])

        text_pred = text_model.predict([url])  # Making prediction using text-based model
        print(text_pred[0])

        feature_confidence = feature_model.predict_proba(features_df)  # Getting confidence scores from feature-based model
        text_confidence = text_model.predict_proba([url])  # Getting confidence scores from text-based model

        # Extracting confidence scores for phishing and legitimate classes
        confidence_phishing_feature = feature_confidence[:, 0] if feature_confidence.shape[1] > 1 else 1 - feature_confidence[:, 0]
        confidence_legitimate_feature = feature_confidence[:, 1] if feature_confidence.shape[1] > 1 else feature_confidence[:, 0]

        confidence_phishing_text = text_confidence[:, 0] if text_confidence.shape[1] > 1 else 1 - text_confidence[:, 0]
        confidence_legitimate_text = text_confidence[:, 1] if text_confidence.shape[1] > 1 else text_confidence[:, 0]

        # Defining weights for each model's confidence score
        weight_feature = 0.6
        weight_text = 0.4

        # Combining confidence scores using weighted average
        combined_confidence_phishing = (weight_feature * confidence_phishing_feature) + (weight_text * confidence_phishing_text)
        combined_confidence_legitimate = (weight_feature * confidence_legitimate_feature) + (weight_text * confidence_legitimate_text)

        # Choosing final prediction based on confidence scores
        if feature_pred == text_pred:
            final_pred = feature_pred
        else:
            final_pred = -1 if combined_confidence_phishing > combined_confidence_legitimate else 1

        # Converting prediction to human-readable format
        result = "Phishing" if final_pred == -1 else "Legitimate"

        # Returning prediction and confidence scores to be rendered in the HTML template
        return render_template('index.html', url=url, result=result)

def get_confidence(model, data):
    """
    Gets confidence score from a model.

    Args:
        model: Trained model object
        data: Input data for prediction

    Returns:
        confidence score
    """
    return model.predict_proba(data)[0]

def choose_prediction(pred1, pred2, confidence):
    """
    Chooses prediction based on confidence scores.

    Args:
        pred1: Prediction from model 1
        pred2: Prediction from model 2
        confidence: Dictionary containing confidence scores

    Returns:
        Final prediction
    """
    if pred1 == pred2:
        return pred1
    else:
        return pred1 if confidence[pred1] > confidence[pred2] else pred2  # Choosing prediction based on confidence scores

if __name__ == '__main__':
    app.run(debug=True)  # Running the Flask application in debug mode
