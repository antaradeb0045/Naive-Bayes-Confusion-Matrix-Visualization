# ==============================
# NAIVE BAYES (BINARY + GUI)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, names=columns)

# Convert to binary classification
# 1 = setosa, 0 = not setosa
data['class'] = data['class'].apply(lambda x: 1 if x == 'Iris-setosa' else 0)

# Features & target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# GUI (Confusion Matrix Heatmap)
# ==============================

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Setosa', 'Setosa'],
            yticklabels=['Not Setosa', 'Setosa'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("2x2 Confusion Matrix (Naive Bayes)")

plt.show()