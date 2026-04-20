# Password Strength Classifier 🔐

A machine learning pipeline that classifies passwords as **weak**, **medium**, or **strong** using feature engineering and multiple ML models.

Built as part of my cybersecurity portfolio to demonstrate the intersection of security knowledge and machine learning.

## Results

| Model | F1 Score (macro) |
|---|---|
| Logistic Regression | 1.0000 |
| Random Forest | 1.0000 |
| XGBoost | 1.0000 |
| SVM | 1.0000 |

> **Note:** The dataset was labeled using three commercial password strength meters (Twitter, Microsoft, and Battle) via Georgia Tech's PARS tool. Only passwords where all three meters agreed on the classification were kept, resulting in ~670k passwords from an original 3M. The high F1 scores reflect the consistency of these labels rather than overfitting.
