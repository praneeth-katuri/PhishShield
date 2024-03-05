# Phishing Detection using Machine Learning

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Flask Deployment with Caching](#flask-deployment-with-caching)
- [Webpage Interface](#webpage-interface)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The project utilizes two separate datasets, each tailored for training a specific machine learning model.

### Dataset for URL-based Model

The dataset used to train the URL-based model.

### Dataset for Text-based Model

The dataset used to train the text-based model.

## Features

The features used by the machine learning models include:

- URL length
- Presence of HTTPS
- Domain age
- IP address reputation
- Presence of suspicious keywords
- Website popularity
- ...

These features are carefully selected to capture different aspects of a website that may indicate whether it is a phishing site or not.

## Machine Learning Models

The repository includes machine learning models trained using various algorithms, including logistic regression, support vector classifier (SVC), random forest, LightGBM, K-nearest neighbors (KNN), and multinomial naive Bayes. These models are trained using scikit-learn and LightGBM libraries.

The training process involves utilizing scikit-learn pipelines, which consist of custom transformers for preprocessing data before feeding it to the models. Grid search with cross-validation is used to tune hyperparameters and optimize model performance.

Each model is evaluated using metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness in distinguishing between phishing and legitimate websites.

## Flask Deployment with Caching

Both machine learning models are deployed using Flask, a lightweight web framework for Python. The Flask app exposes endpoints to make predictions using the trained models. Additionally, caching to disk is implemented to improve performance by storing results of previous predictions.

## Webpage Interface

The web interface is built using HTML, CSS, and Bootstrap to provide a user-friendly experience. Users can input a URL and receive predictions on whether it is a phishing website or not.

![Input](https://i.imgur.com/example_input.png)
*Input webpage interface*

![Output](https://i.imgur.com/example_output.png)
*Output webpage interface*

## Usage

To use the phishing detection models, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/phishing-detection.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the provided scripts to train the models, perform grid search, and deploy the Flask app.

4. Use the Flask app to make predictions on new URLs.

## Results

The performance of the phishing detection models is evaluated using metrics such as accuracy, precision, recall, and F1-score. The results demonstrate the effectiveness of each model in distinguishing between phishing and legitimate websites.

## Contributing

Contributions to this project are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.