import joblib
import pandas as pd
from pathlib import Path
from features import extract_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
LABEL_MAP = {0: "Weak", 1: "Medium", 2: "Strong"}

def predict_strength(password: str) -> str:
    model = joblib.load(MODEL_PATH)
    features = extract_features(password)
    X = pd.DataFrame([features])
    pred = model.predict(X)[0]
    return LABEL_MAP[int(pred)]

if __name__ == "__main__":
    test_passwords = [
        "123456",
        "Password1",
        "X#9kLm$2pQ!rTv",
        "qwerty",
        "C0rr3ct-H0rs3-B@tt3ry",
    ]
    print("Password Strength Predictions\n" + "="*40)
    for pw in test_passwords:
        print(f"{pw:<30} → {predict_strength(pw)}")