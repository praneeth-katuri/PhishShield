from flask import Blueprint, render_template, request, make_response, redirect
import requests
from utils.config import Config
from app.app import csrf

recaptcha_bp = Blueprint('recaptcha', __name__)

@recaptcha_bp.route('/verify_recaptcha', methods=['GET', 'POST'])
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
                'secret': Config.RECAPTCHA_SECRET_KEY,
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
    
    return render_template('verification.html', RECAPTCHA_SITE_KEY=Config.RECAPTCHA_SITE_KEY)