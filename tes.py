import joblib
feature_model = joblib.load("models/feature_model.joblib")
url = "https://snowy-cloud-15ee.exka.workers.dev/tspd/login/loginhelp"
feature_pred = feature_model.predict(url)
print(feature_pred)
