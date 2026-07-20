import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def train_nlp_fraud_classifier():
    print("==================================================================")
    print("[TRAINING] SHOPEE SENTINEL - NLP DECEPTIVE TEXT MODEL TRAINING SCRIPT")
    print("==================================================================")
    print()

    # 1. Locate dataset
    materials_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Plan", "materials")
    dataset_path = os.path.join(materials_dir, "fake_reviews.csv")

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset file not found at: {dataset_path}")
        return

    print(f"[1/5] Loading training dataset: {dataset_path} ...")
    df = pd.read_csv(dataset_path)
    print(f"      Total samples in dataset: {len(df):,}")
    print(f"      Columns found: {list(df.columns)}")
    print(f"      Class distribution:\n{df['label'].value_counts().to_string()}")
    print()

    # Clean text data
    df = df.dropna(subset=['text', 'label'])
    X = df['text'].astype(str)
    y = df['label'].astype(int)

    # 2. Split Train / Test
    print("[2/5] Splitting data into 80% Training set and 20% Test evaluation set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train set size: {len(X_train):,} samples")
    print(f"      Test set size:  {len(X_test):,} samples")
    print()

    # 3. Text Vectorization (TF-IDF)
    print("[3/5] Extracting TF-IDF n-gram text features...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words='english',
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train Classifier
    print("[4/5] Training Logistic Regression classification model...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate Model
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"      [SUCCESS] Training Complete! Model Accuracy on Test Set: {acc * 100:.2f}%")
    print("\nDetailed Classification Metrics:")
    print(classification_report(y_test, y_pred, target_names=["Genuine (0)", "Deceptive/Fake (1)"]))

    # Save model artifacts
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "saved_models")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "nlp_fraud_model.pkl")
    vec_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"[5/5] Saved trained model to: {model_path}")
    print(f"      Saved vectorizer to:    {vec_path}")
    print()

    # Demonstration predictions
    print("==================================================================")
    print("[DEMO] LIVE MODEL PREDICTION DEMO")
    print("==================================================================")
    sample_texts = [
        "100% original AAA quality factory direct price no warranty super cheap",
        "Great quality product, fast delivery and item arrived in original sealed packaging.",
        "Replica copy same as original act fast limited time unclaimed box"
    ]

    sample_vecs = vectorizer.transform(sample_texts)
    probs = clf.predict_proba(sample_vecs)[:, 1]

    for text, prob in zip(sample_texts, probs):
        label = "RED (DECEPTIVE/FAKE)" if prob > 0.5 else "GREEN (GENUINE)"
        print(f"Text:   '{text}'")
        print(f"Result: {label} (Deceptive Probability: {prob*100:.1f}%)\n")

if __name__ == "__main__":
    train_nlp_fraud_classifier()
