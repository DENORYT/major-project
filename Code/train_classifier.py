"""
ISL Classifier Training Script
===============================
Trains a Random Forest classifier on collected hand landmark data.

Usage:
    python train_classifier.py

Prerequisites:
    Run collect_data.py first to create landmark_data.csv

Output:
    isl_classifier.pkl — trained model + label mapping
"""

import os, pickle, csv
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

DATA_FILE = os.path.join(script_dir, 'landmark_data.csv')
MODEL_FILE = os.path.join(script_dir, 'isl_classifier.pkl')


def main():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found!")
        print("Run collect_data.py first to collect training data.")
        return

    # Load data
    labels = []
    features = []

    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            labels.append(row[0])
            features.append([float(x) for x in row[1:]])

    X = np.array(features)
    y_labels = np.array(labels)

    # Create label encoding
    unique_labels = sorted(set(y_labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    y = np.array([label_to_int[label] for label in y_labels])

    print(f"Dataset: {len(X)} samples, {len(unique_labels)} classes")
    print(f"Classes: {unique_labels}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining: {len(X_train)} samples")
    print(f"Testing:  {len(X_test)} samples")

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.1f}%")
    print("\nClassification Report:")
    target_names = [int_to_label[i] for i in range(len(unique_labels))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save model
    model_data = {
        'model': clf,
        'labels': int_to_label,
        'label_to_int': label_to_int,
        'accuracy': accuracy,
        'n_classes': len(unique_labels),
        'n_samples': len(X),
    }

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {MODEL_FILE}")
    print(f"Restart the web app to use the new model!")


if __name__ == '__main__':
    main()
