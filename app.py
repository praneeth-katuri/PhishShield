import joblib
import requests
import os
from flask import Flask, render_template, request, make_response, redirect
from flask_caching import Cache
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.secret_key = 'e05ffb424fdc73e5d591ce023a72b88e'
csrf = CSRFProtect(app)

# Configure Flask-Caching to use on-disk caching
cache_dir = os.path.join(app.root_path, '.cache')
cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': cache_dir})

# Loading the trained models
text_model = joblib.load("models/text_model.joblib")
feature_model = joblib.load("models/feature_model.joblib")

# Configure reCAPTCHA keys
RECAPTCHA_SITE_KEY = '6Lc72ZopAAAAAFfq3Rn3xoJj_dHBfejI7CNXzoBO'
RECAPTCHA_SECRET_KEY = '6Lc72ZopAAAAANE7ep2VVkALwBBcIM1nMXN6euDA'

@app.route('/verify_recaptcha', methods=['GET', 'POST'])
@csrf.exempt
def verify_recaptcha():
    if request.method == 'POST':
        token = request.form.get('g-recaptcha-response')
        if not token:
            return render_template('verification.html', error='Please complete the reCAPTCHA.')
        
        # Verify reCAPTCHA token with Google
        response = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': RECAPTCHA_SECRET_KEY,
                'response': token
            }
        )

        if response.ok:
            result = response.json()
            if result.get('success'):
                # reCAPTCHA verification successful, set a cookie to indicate verification
                resp = make_response(redirect('/'))
                resp.set_cookie('recaptcha_verified', 'true')
                return resp
            else:
                return render_template('verification.html', error='reCAPTCHA verification failed.')
        else:
            return render_template('verification.html', error='Failed to verify reCAPTCHA. Please try again later.')
    
    return render_template('verification.html', RECAPTCHA_SITE_KEY=RECAPTCHA_SITE_KEY)

# Defining route for phishing detection
@app.route('/', methods=['GET','POST'])
@csrf.exempt
def detect_phishing():
    if request.cookies.get('recaptcha_verified') == 'true':
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

                final_pred = -1 if combined_confidence_phishing > combined_confidence_legitimate else 1

            result = "Phishing" if final_pred == -1 else "Legitimate"
            cache.set(url, render_template('index.html', url=url, result=result), timeout=3600)  # Cache for 1 hour (3600 seconds)

            # Returning prediction and confidence scores to be rendered in the HTML template
            return render_template('index.html', url=url, result=result)
        else:
            return render_template('index.html')
    else:
        return redirect('/verify_recaptcha')


if __name__ == '__main__':
    app.run(debug=True)
