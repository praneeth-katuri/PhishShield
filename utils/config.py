# backend/utils/config.py
from dotenv import load_dotenv
import os

load_dotenv()

FLASK_SECRET_KEY = os.getenv("SECRET_KEY")
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")
RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY")
