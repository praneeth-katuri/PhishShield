from flask import Flask
from flask_caching import Cache
from utils.config import Config
from app.csrf_init import csrf 
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
# Initialize CSRF protection
csrf.init_app(app)

cache_dir = os.path.join(app.root_path, '.cache')
cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': cache_dir})

# Import and register the blueprints for routes
from app.routes.recaptcha import recaptcha_bp
from app.routes.detect import detect_bp

app.register_blueprint(recaptcha_bp)
app.register_blueprint(detect_bp)


if __name__ == '__main__':
    app.run(debug=True)
