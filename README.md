# Password Strength Classifier 

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

## Features Extracted

| Feature | Description |
|---|---|
| `length` | Total password length |
| `entropy` | Shannon entropy — measures unpredictability |
| `num_uppercase` | Number of uppercase letters |
| `num_lowercase` | Number of lowercase letters |
| `num_digits` | Number of digits |
| `num_special` | Number of special characters |
| `unique_char_ratio` | Unique characters / total length |
| `has_consecutive` | Detects sequences like `aaa`, `123`, `cba` |
| `is_common` | Flags passwords from known common password lists |

## Installation

```bash
git clone https://github.com/rcrzlbrd/password-strength-classifier.git
cd password-strength-classifier
python -m venv .venv
source .venv/bin/activate.fish  # or source .venv/bin/activate on bash
pip install -r requirements.txt
```

## Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset) and place it as `data/data.csv`.

## Usage

Train the model:
```bash
cd src
python train.py
```

Predict password strength:
```bash
python predict.py
```
## Tech Stack

- **Python 3.14**
- **scikit-learn** — model training and evaluation
- **XGBoost** — gradient boosting classifier
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — results visualization
- **joblib** — model persistence

## Key Learnings

- Designed security-informed features using domain knowledge (entropy, consecutive patterns, common password detection)
- Benchmarked multiple ML models using F1 macro score to handle class imbalance
- Built a reusable prediction pipeline that loads a trained model and classifies new passwords in real time
- Investigated dataset labeling methodology to explain perfect classification scores

## Author

**Ramiro Cruz Labrada** — [GitHub](https://github.com/rcrzlbrd)
