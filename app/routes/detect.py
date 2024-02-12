from flask import Blueprint, request, render_template, redirect
from app.app import cache, csrf
from app.models import predict_url
from utils.config import Config

detect_bp = Blueprint('detect', __name__)

@detect_bp.route('/', methods=['GET', 'POST'])
@csrf.exempt
def detect_phishing():
    if request.cookies.get('recaptcha_verified') != 'true':
        return redirect('/verify_recaptcha')

    if request.method == 'POST':
        url = request.form.get("url")

        # Check cache first
        cached_result = cache.get(url)
        if cached_result:
            return cached_result

        # Predict
        final_pred = predict_url(url)
        result = "Phishing" if final_pred == -1 else "Legitimate"

        # Cache the rendered result
        rendered = render_template('index.html', url=url, result=result)
        cache.set(url, rendered, timeout=3600)  # Cache for 1 hour

        return rendered

    return render_template('index.html')
