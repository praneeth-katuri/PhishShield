# PhishShield

PhishShield is a phishing website detector using two ML models (feature-based and text-based).

## Datasets

The project utilizes two separate datasets, each tailored for training a specific machine learning model.

### Dataset for Feature-based Model

- [The dataset used to train the feature-based model](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector)

### Dataset for Text-based Model

- [The dataset used to train the text-based model](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls)

## Models

- **Feature-based:** Makes prediction based on 29 URL features extracted from the URL.

- **Text-based:** Makes predition by analyzing URL text, words used in URL.

- Both models use pipelines, transformers, and hyperparameter tuning with grid search.

## Deployment

- **Backend:** Flask app serving prediction endpoints. Disk caching for improving speed.

- **Frontend:** Simple HTML/CSS/Bootstrap UI. Enter a URL, get prediction.

## Webpage Interface

<div align="center">
    <img src="screenshots/phishing.gif" alt="Image 1" width="600" height="300" style="margin-right: 20px;">
    <img src="screenshots/legitimate.gif" alt="Image 2" width="600" height="300" style="margin-right: 20px;">
</div>

## Usage

To use the PhishShield, follow these steps:

1. Clone the repository:

   ```
   git clone --depth=1 https://github.com/praneeth-katuri/PhishShield.git
   ```

2. Install the required dependencies:

   Python Version: `3.12.3`

   ```
   pip install -r requirements.txt
   ```

3. Run the NLTK setup script:

   ```
   python utils/setup_nltk.py
   ```

4. Edit `.env` file and enter your [reCAPTCHA Keys](https://developers.google.com/recaptcha/intro) and `Flask Secret Key`

   To generate `Flask Secret Key` run the below code in terminal and copy the Output key obtained in `.env` file

   ```
   python -c 'import secrets; print(secrets.token_hex(16))'
   ```

5. To start the Flask application, run the following command in your terminal:

   ```
   python run.py
   ```

6. To access the webpage interface, open `http://127.0.0.1:5000` in your web browser.

## Results

**Metrics Evaluated**: accuracy, precision, recall, F1-score.

### Feature-based Model

<div align="center">
    <img src="screenshots/result1.png" alt="Image 1" width="900" height="350" style="margin-right: 20px;">
</div>

### Text-based Model

<div align="center">
    <img src="screenshots/result2.png" alt="Image 2" width="600" height="350" style="margin-right: 20px;">
</div>

## Contributing

Contributions to this project are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
