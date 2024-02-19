import os
import joblib

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

text_model_path = os.path.join(MODELS_DIR, 'text_model.joblib')
feature_model_path = os.path.join(MODELS_DIR, 'feature_model.joblib')

text_model = joblib.load(text_model_path)
feature_model = joblib.load(feature_model_path)

def predict_url(url):
    # Making prediction using feature-based model
    feature_pred = feature_model.predict([url])
    print(feature_pred[0])

    # Making prediction using text-based model
    text_pred = text_model.predict([url])
    print(text_pred[0])

    # Choosing final prediction based on confidence scores
    if feature_pred == text_pred:
        return feature_pred
    else:
        # Getting confidence scores from feature-based model
        feature_confidence = feature_model.predict_proba([url])
        print(feature_confidence)

        # Getting confidence scores from text-based model
        text_confidence = text_model.predict_proba([url])
        print(text_confidence)

        # Extracting confidence scores for phishing and legitimate classes
        confidence_phishing_feature = feature_confidence[:,0] if feature_confidence.shape[1] > 1 else 1 - feature_confidence[:, 0]

        confidence_legitimate_feature = feature_confidence[:,1] if feature_confidence.shape[1] > 1 else feature_confidence[:, 0]

        confidence_phishing_text = text_confidence[:,0] if text_confidence.shape[1] > 1 else 1 - text_confidence[:, 0]

        confidence_legitimate_text = text_confidence[:,1] if text_confidence.shape[1] > 1 else text_confidence[:, 0]

        # Defining weights for each model's confidence score
        weight_feature = 0.6
        weight_text = 0.4

        # Combining confidence scores using weighted average
        combined_confidence_phishing = (weight_feature * confidence_phishing_feature) + (weight_text * confidence_phishing_text)
        combined_confidence_legitimate = (weight_feature * confidence_legitimate_feature) + (weight_text * confidence_legitimate_text)

        return -1 if combined_confidence_phishing > combined_confidence_legitimate else 1