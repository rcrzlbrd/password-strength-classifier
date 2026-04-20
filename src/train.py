import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

LABEL_MAP = {0: "weak", 1: "medium", 2: "strong"}

def load_and_featurize(path: Path):
    print("Cargando dataset...")
    df = pd.read_csv(path, on_bad_lines="skip")
    df.dropna(subset=["password", "strength"], inplace=True)
    df["password"] = df["password"].astype(str)

    print("Extrayendo features...")
    features = [extract_features(pw) for pw in tqdm(df["password"])]
    X = pd.DataFrame(features)
    y = df["strength"].astype(int)
    return X, y

def benchmark_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss"),
        "SVM": SVC(kernel="rbf", random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        results[name] = {"model": model, "f1_macro": round(f1, 4), "y_pred": y_pred}
        print(f"  F1 macro: {f1:.4f}")

    return results

def save_results(best_name, best_model, y_test, y_pred, results):
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print(f"Modelo guardado: {best_name}")

    metrics = {name: {"f1_macro": v["f1_macro"]} for name, v in results.items()}
    metrics["best_model"] = best_name
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_MAP.values(),
                yticklabels=LABEL_MAP.values())
    plt.title(f"Confusion Matrix — {best_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        feat_names = list(extract_features("test").keys())
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        fi_df.sort_values("importance", ascending=True, inplace=True)
        plt.figure(figsize=(8, 5))
        plt.barh(fi_df["feature"], fi_df["importance"], color="steelblue")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
        plt.close()

def main():
    X, y = load_and_featurize(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = benchmark_models(X_train, X_test, y_train, y_test)

    #best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_name = "RandomForest"
    best = results[best_name]
    print(f"\nMejor modelo: {best_name} (F1={best['f1_macro']})")
    print(classification_report(y_test, best["y_pred"],
                                target_names=list(LABEL_MAP.values())))

    save_results(best_name, best["model"], y_test, best["y_pred"], results)


if __name__ == "__main__":
    main()