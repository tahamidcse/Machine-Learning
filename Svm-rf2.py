# ============================================
# Question 1: Random Forest vs SVM Classification
# Dataset: Wine Quality (3 wine types, 13 features)
# ============================================

# 1. Load the dataset
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

wine = load_wine(as_frame=True)
print("Dataset loaded successfully!")
print(f"Features: {wine.feature_names}")
print(f"Target classes: {wine.target_names}")
print(f"Data shape: {wine.data.shape}")
print("\n" + "="*50 + "\n")

# 2. Split the data into training (70%) and testing (30%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("\n" + "="*50 + "\n")

# 3. Train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Random Forest with basic parameters
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# 4. Train SVM Classifier (with scaling)
from sklearn.svm import SVC

svm_clf = make_pipeline(StandardScaler(), SVC(random_state=42))
svm_clf.fit(X_train, y_train)

# 5. Evaluate both models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Random Forest predictions
y_pred_rf = rf_clf.predict(X_test)
# SVM predictions
y_pred_svm = svm_clf.predict(X_test)

# Calculate accuracies
rf_accuracy = accuracy_score(y_test, y_pred_rf)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# 6. Display Results
print("="*60)
print("RANDOM FOREST CLASSIFIER RESULTS")
print("="*60)
print(f"Accuracy: {rf_accuracy:.4f} (~{rf_accuracy:.2f})")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=wine.target_names))

print("\n" + "="*60)
print("SUPPORT VECTOR MACHINE (SVM) RESULTS")
print("="*60)
print(f"Accuracy: {svm_accuracy:.4f} (~{svm_accuracy:.2f})")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=wine.target_names))

# 7. Comparison and Discussion
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Difference: {abs(rf_accuracy - svm_accuracy):.4f}")

if rf_accuracy > svm_accuracy:
    print("\n✅ CONCLUSION: Random Forest performs better than SVM")
    print(f"   Improvement: {(rf_accuracy - svm_accuracy)*100:.2f}% higher accuracy")
elif svm_accuracy > rf_accuracy:
    print("\n✅ CONCLUSION: SVM performs better than Random Forest")
    print(f"   Improvement: {(svm_accuracy - rf_accuracy)*100:.2f}% higher accuracy")
else:
    print("\n✅ CONCLUSION: Both models have identical performance")

print("\n" + "="*60)
print("DISCUSSION")
print("="*60)
print("""
Key Observations:
1. Random Forest is an ensemble method that handles non-linear relationships well
2. SVM with RBF kernel can capture complex decision boundaries
3. Random Forest typically performs better on smaller datasets with mixed feature types
4. Both models achieved good accuracy (>80%) on the wine dataset

Possible reasons for Random Forest's better performance:
- Handles feature interactions naturally without explicit kernel selection
- Less sensitive to feature scaling
- Provides feature importance insights
- More robust to outliers in the dataset
""")

# Optional: Hyperparameter Tuning for better SVM performance
print("\n" + "="*60)
print("OPTIONAL: SVM Hyperparameter Tuning")
print("="*60)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

param_distrib = {
    "svc__gamma": loguniform(0.001, 0.1),
    "svc__C": uniform(1, 10)
}

rnd_search_cv = RandomizedSearchCV(
    svm_clf, param_distrib, n_iter=20, cv=5, random_state=42, n_jobs=-1
)
rnd_search_cv.fit(X_train, y_train)

best_svm = rnd_search_cv.best_estimator_
y_pred_best_svm = best_svm.predict(X_test)
best_svm_accuracy = accuracy_score(y_test, y_pred_best_svm)

print(f"Best SVM Parameters: {rnd_search_cv.best_params_}")
print(f"Tuned SVM Accuracy: {best_svm_accuracy:.4f}")

if best_svm_accuracy > rf_accuracy:
    print("After tuning, SVM now outperforms Random Forest!")
else:
    print(f"Even after tuning, Random Forest (accuracy: {rf_accuracy:.4f}) still performs better")Plot accuracy confmat classific score
