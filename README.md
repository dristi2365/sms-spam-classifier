# SMS Spam Classifier

A machine learning project that classifies SMS messages as **Spam** or **Ham (not spam)** using NLP techniques.

## Overview

This project demonstrates a classic NLP pipeline — from raw text to a trained classifier — using a real-world SMS dataset. It applies TF-IDF vectorization and a Multinomial Naive Bayes model to achieve high classification accuracy.

## How It Works

1. Downloads the **SMS Spam Collection dataset** (~5,500 messages) directly from UCI ML Repository
2. Preprocesses and splits the data into training and test sets
3. Builds a pipeline with **TF-IDF vectorization** + **Multinomial Naive Bayes**
4. Evaluates using accuracy, precision, recall, and F1-score
5. Accepts user input for real-time message prediction

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~97–99% |
| Classifier | Multinomial Naive Bayes |
| Vectorizer | TF-IDF |
| Dataset | UCI SMS Spam Collection (5,574 messages) |

## How to Run

1. Make sure Python is installed
2. Install dependencies:
```bash
   pip install pandas scikit-learn
```
3. Run the script:
```bash
   python sms_spam_classifier.py
```
4. Enter any SMS message when prompted to get a prediction

## Files

| File | Description |
|------|-------------|
| `sms_spam_classifier.py` | Main script — data loading, training, evaluation, prediction |
| `README.md` | Project documentation |

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, Scikit-learn
- **Model:** Multinomial Naive Bayes
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
