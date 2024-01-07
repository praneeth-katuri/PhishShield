from flask import Flask
from utils.config import FLASK_SECRET_KEY

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY


@app.route("/health")
def health():
    return {"status": "OK"}, 200


if __name__ == "__main__":
    app.run(debug=True)
