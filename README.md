# Customer Churn Prediction

This project is a machine learning solution for predicting customer churn using the Telco Customer Churn dataset.

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave a service.  
In this project, I built a churn prediction pipeline using multiple classification models and evaluated them with AUC score.

## Dataset

- **Dataset Name:** Telco Customer Churn
- **Source:** Kaggle / Telco Customer Churn dataset
- **Target Column:** `Churn`

## Workflow

1. Load the dataset
2. Inspect missing values and basic statistics
3. Remove unnecessary columns
4. Data visualization
5. Encode categorical and binary features
6. Clean numeric columns
7. Split data into training and validation sets
8. Train multiple machine learning models
9. Evaluate models using ROC AUC score

## Models Used

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree Classifier

## Evaluation Metric

The models are evaluated using **ROC AUC score**, which is suitable for binary classification problems like churn prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Goal

The goal of this project is to provide a clean and practical machine learning workflow for customer churn prediction.

## Notes

- The project focuses on preprocessing, model training, and evaluation.
- It can be extended with feature engineering, hyperparameter tuning, and model comparison improvements.

## Repository Structure
```bash
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_prediction.py
└── README.md