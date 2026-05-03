#  train_model.py   Train & save Logistic Regression + Decision Tree

import numpy as np
import pandas as pd
import joblib, json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#  1. LOAD DATASET FROM CSV (UPDATED) 
df = pd.read_csv("lung_cancer_dataset.csv")

# Clean column names (important for your dataset)
df.columns = df.columns.str.strip().str.upper()

# Convert categorical values to numeric
df = df.replace({
    "M": 1,
    "F": 0,
    "YES": 1,
    "NO": 0
})

# Ensure all values are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Handle missing values (safe fix)
df = df.fillna(df.median(numeric_only=True))

# Ensure target column exists
if "LUNG_CANCER" not in df.columns:
    raise ValueError("LUNG_CANCER column not found in dataset")

df["LUNG_CANCER"] = df["LUNG_CANCER"].astype(int)

print(f"Dataset: {len(df)} records, {df['LUNG_CANCER'].sum()} positive cases")

#  2. Split & scale 
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

#  3. Train Logistic Regression 
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_lr = lr.predict(X_test_s)

#  4. Train Decision Tree 
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_dt = dt.predict(X_test)

#  5. Save models & metrics 
joblib.dump(lr,     'ml_model/logistic_regression.pkl')
joblib.dump(dt,     'ml_model/decision_tree.pkl')
joblib.dump(scaler, 'ml_model/scaler.pkl')

meta = {
    'feature_names': list(X.columns),
    'logistic_regression': {
        'accuracy':  round(accuracy_score(y_test, y_lr), 4),
        'precision': round(precision_score(y_test, y_lr), 4),
        'recall':    round(recall_score(y_test, y_lr), 4),
        'f1':        round(f1_score(y_test, y_lr), 4),
        'cv_mean':   round(cross_val_score(lr, X_train_s, y_train, cv=5).mean(), 4),
        'confusion_matrix': confusion_matrix(y_test, y_lr).tolist(),
    },
    'decision_tree': {
        'accuracy':  round(accuracy_score(y_test, y_dt), 4),
        'precision': round(precision_score(y_test, y_dt), 4),
        'recall':    round(recall_score(y_test, y_dt), 4),
        'f1':        round(f1_score(y_test, y_dt), 4),
        'cv_mean':   round(cross_val_score(dt, X_train, y_train, cv=5).mean(), 4),
        'confusion_matrix': confusion_matrix(y_test, y_dt).tolist(),
    },
}

with open('ml_model/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n  Models saved!")
print(f"   Logistic Regression accuracy: {meta['logistic_regression']['accuracy']}")
print(f"   Decision Tree accuracy:       {meta['decision_tree']['accuracy']}")


class_names = ["NO LUNG CANCER", "LUNG CANCER"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Logistic Regression
cm1 = confusion_matrix(y_test, lr.predict(X_test_s))
ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=class_names)\
    .plot(ax=ax[0], colorbar=False)

ax[0].set_title("Logistic Regression")

# Decision Tree
cm2 = confusion_matrix(y_test, dt.predict(X_test))
ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=class_names)\
    .plot(ax=ax[1], colorbar=False)

ax[1].set_title("Decision Tree")

plt.tight_layout()
plt.show()