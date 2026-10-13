"""SMS spam classifier: TF-IDF + Naive Bayes, tuned and evaluated properly.

Usage:
    python spam_classifier.py
    python spam_classifier.py train
    python spam_classifier.py predict "Claim your free lottery prize now!" "See you at 6?"
"""

import argparse
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_PATH = Path("data/sms.tsv")
MODEL_PATH = Path("spam_model.joblib")
RANDOM_STATE = 42

# Settings the grid search will try.
PARAM_GRID = {
    "tfidf__stop_words": [None, "english"],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "nb__alpha": [0.01, 0.1, 0.5, 1.0],
}


def load_data():
    """Load the dataset, downloading it once and caching it in data/."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        print("Downloading dataset...")
        with urllib.request.urlopen(DATA_URL, timeout=15) as response:
            DATA_PATH.write_bytes(response.read())
    return pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"])


def build_pipeline():
    """Vectorizer + model in ONE object, so the vectorizer only ever learns
    from training data (no data leakage) and both are saved together."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("nb", MultinomialNB()),
    ])


def train():
    df = load_data()
    print(f"Loaded {len(df)} messages:\n{df['label'].value_counts().to_string()}\n")

    # stratify keeps the spam/ham ratio the same
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )

    
    search = GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        scoring=make_scorer(f1_score, pos_label="spam"),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print("Best settings:", search.best_params_)
    print(f"Cross-validated spam F1: {search.best_score_:.4f}\n")

    
    model = search.best_estimator_
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, digits=4))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=["ham", "spam"]),
        index=["actual ham", "actual spam"],
        columns=["pred ham", "pred spam"],
    ))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def predict(messages):
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run: python spam_classifier.py train")

    model = joblib.load(MODEL_PATH)
    spam_column = list(model.classes_).index("spam")
    labels = model.predict(messages)
    probabilities = model.predict_proba(messages)

    for message, label, probs in zip(messages, labels, probabilities):
        print(f"{label.upper():4} ({probs[spam_column]:6.1%} spam)  {message}")


def main():
    parser = argparse.ArgumentParser(description="SMS spam classifier")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", help="tune, evaluate and save the model")
    predict_parser = subparsers.add_parser("predict", help="classify messages")
    predict_parser.add_argument("messages", nargs="+", help="one or more messages in quotes")
    args = parser.parse_args()

    if args.command == "predict":
        predict(args.messages)
    else: 
        train()


if __name__ == "__main__":
    main()
