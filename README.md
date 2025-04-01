# Sentiment Analysis using Machine Learning

This project performs sentiment analysis on IMDB movie reviews using Natural Language Processing (NLP) and Machine Learning models. The goal is to classify movie reviews as either positive or negative.

## Features
- Preprocesses text data by cleaning, tokenizing, and lemmatizing reviews.
- Uses three classification models: Logistic Regression, Naive Bayes, and Support Vector Machine (SVM).
- Implements TF-IDF vectorization for feature extraction.
- Evaluates model performance using accuracy, precision, recall, and F1-score.
- Performs cross-validation for robust evaluation.

## Installation
### Clone the repository:
```sh
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```
### Install the required dependencies:
```sh
pip install -r requirements.txt
```
### Download necessary NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Dataset
The dataset consists of IMDB movie reviews with labeled sentiments (positive or negative). The data is loaded using Pandas and preprocessed for machine learning models.

## Project Workflow
1. **Import Dependencies** - Load required libraries for data processing, NLP, and machine learning.
2. **Download NLTK Resources** - Ensure required linguistic datasets are available.
3. **Load Dataset** - Read and explore the dataset.
4. **Preprocess Text** - Clean and transform the text data.
5. **Prepare Data for ML** - Encode labels and split into training & testing sets.
6. **Train Models** - Train Logistic Regression, Naive Bayes, and SVM classifiers.
7. **Evaluate Performance** - Use classification reports and confusion matrices.
8. **Cross-Validation** - Ensure model robustness using Stratified K-Fold cross-validation.
9. **Make Predictions** - Predict sentiment of new reviews using the best-performing model.

## Model Performance
| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | 88%      |
| Naive Bayes           | 85%      |
| Support Vector Machine| 88%      |

## LSTM Model for Sentiment Analysis
### Overview
In addition to traditional machine learning models, this project implements a Long Short-Term Memory (LSTM) network, a type of recurrent neural network (RNN) designed to handle sequential data such as text. LSTMs are effective for sentiment analysis because they can retain information from earlier words in a sentence, capturing long-range dependencies.

### LSTM Architecture
- **Embedding Layer**: Converts words into dense vector representations.
- **LSTM Layer**: Processes sequences of word embeddings while maintaining long-term dependencies.
- **Fully Connected Layer**: Outputs a single value indicating sentiment probability.
- **Sigmoid Activation**: Maps the output to a probability between 0 and 1 for binary classification.

### Training Process
1. **Preprocessing**: Text data is cleaned, tokenized, and converted into sequences using a vocabulary index.
2. **Padding**: Sequences are padded to a fixed length to maintain uniform input size.
3. **Batch Processing**: Data is loaded into batches using PyTorch's DataLoader for efficient training.
4. **Optimization**: The model is trained using the Adam optimizer and binary cross-entropy loss.
5. **Evaluation**: Model performance is assessed on a test dataset, and accuracy is computed.

### Usage
To train and evaluate the models, run:
```sh
python main.py
```
To predict sentiment for a custom review, use:
```python
predict_sentiment("The movie was fantastic and thrilling!")
```

## Future Improvements
- Implement deep learning models such as LSTMs or Transformers.
- Enhance feature extraction with word embeddings.
- Deploy as a web application for real-time sentiment analysis.
