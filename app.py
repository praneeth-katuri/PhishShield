import joblib
import os
from flask import Flask, render_template, request
from flask_caching import Cache
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)  # Creating Flask application instance
csrf = CSRFProtect(app)  # Adding CSRF protection to the app
# Configure Flask-Caching to use on-disk caching
cache_dir = os.path.join(app.root_path, '.cache')
cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': cache_dir})
# Loading the trained models
# Loading text-based phishing detection model
text_model = joblib.load("models/text_model.joblib")
# Loading feature-based phishing detection model
feature_model = joblib.load("models/feature_model.joblib")


@app.route('/')  # Defining route for the home page
def home():
    """
    Renders the home page.

    Returns:
        rendered HTML template
    """
    return render_template('index.html')  # Rendering the HTML template for the home page


# Defining route for phishing detection
@app.route('/detect', methods=['POST'])
@csrf.exempt  # Exempting CSRF protection for this route
def detect_phishing():
    """
    Detects phishing URLs.

    Returns:
        rendered HTML template with the prediction result
    """
    if request.method == 'POST':
        url = request.form["url"]  # Getting URL from the form data
        # Check if the result is cached
        cached_result = cache.get(url)
        if cached_result:
            return cached_result

        # Making prediction using feature-based model
        feature_pred = feature_model.predict([url])
        print(feature_pred[0])

        # Making prediction using text-based model
        text_pred = text_model.predict([url])
        print(text_pred[0])

        # Choosing final prediction based on confidence scores
        if feature_pred == text_pred:
            final_pred = feature_pred
        else:
            # Getting confidence scores from feature-based model
            feature_confidence = feature_model.predict_proba([url])
            print(feature_confidence)
            # Getting confidence scores from text-based model
            text_confidence = text_model.predict_proba([url])
            print(text_confidence)
            # Extracting confidence scores for phishing and legitimate classes
            confidence_phishing_feature = feature_confidence[:,
                                                             0] if feature_confidence.shape[1] > 1 else 1 - feature_confidence[:, 0]
            print(confidence_phishing_feature)
            confidence_legitimate_feature = feature_confidence[:,
                                                               1] if feature_confidence.shape[1] > 1 else feature_confidence[:, 0]
            print(confidence_legitimate_feature)
            confidence_phishing_text = text_confidence[:,
                                                       0] if text_confidence.shape[1] > 1 else 1 - text_confidence[:, 0]
            print(confidence_phishing_text)
            confidence_legitimate_text = text_confidence[:,
                                                         1] if text_confidence.shape[1] > 1 else text_confidence[:, 0]
            print(confidence_legitimate_text)
            # Defining weights for each model's confidence score
            weight_feature = 0.6
            weight_text = 0.4

            # Combining confidence scores using weighted average7i
            combined_confidence_phishing = (
                weight_feature * confidence_phishing_feature) + (weight_text * confidence_phishing_text)
            combined_confidence_legitimate = (
                weight_feature * confidence_legitimate_feature) + (weight_text * confidence_legitimate_text)

            final_pred = -1 if combined_confidence_phishing > combined_confidence_legitimate else 1

        # Converting prediction to human-readable format
        result = "Phishing" if final_pred == -1 else "Legitimate"
        cache.set(url, render_template('index.html', url=url, result=result),
                  timeout=3600)  # Cache for 1 hour (3600 seconds)
        # Returning prediction and confidence scores to be rendered in the HTML template
        return render_template('index.html', url=url, result=result)


if __name__ == '__main__':
    app.run(debug=True)  # Running the Flask application in debug mode
