# ============================================
# Question 1: Random Forest vs SVM Classification
# Dataset: Wine Quality (3 wine types, 13 features)
# ============================================

# 1. Load the dataset
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Calculate OOB error for different numbers of trees
n_trees_range = range(10, 201)
oob_errors = []
for n in n_trees_range:
    rf_temp = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf_temp.fit(X_train, y_train)
    oob_errors.append(1 - rf_temp.oob_score_)

# Print results at key intervals
print("OOB Error by Number of Trees:")
for i, error in enumerate(oob_errors, start=10):
    if i % 10 == 0:
        print(f"Trees: {i:3d}, OOB Error: {error:.4f}")

# Plot OOB error
plt.figure(figsize=(10, 6))
plt.plot(n_trees_range, oob_errors)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error')
plt.title('Random Forest - OOB Error vs Number of Trees')
plt.grid(True)
plt.show()

# 4. Train SVM Classifier (with scaling)
from sklearn.svm import SVC

svm_clf = make_pipeline(StandardScaler(), SVC(random_state=42, probability=True))
svm_clf.fit(X_train, y_train)

# 5. Evaluate both models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Random Forest predictions
y_pred_rf = rf_clf.predict(X_test)
y_pred_proba_rf = rf_clf.predict_proba(X_test)

# SVM predictions
y_pred_svm = svm_clf.predict(X_test)
y_pred_proba_svm = svm_clf.predict_proba(X_test)

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

# 7. Confusion Matrix Visualization - Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_rf, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()

plt.xticks(range(len(wine.target_names)), wine.target_names, rotation=45)
plt.yticks(range(len(wine.target_names)), wine.target_names)

# Add text annotations
for i in range(conf_matrix_rf.shape[0]):
    for j in range(conf_matrix_rf.shape[1]):
        plt.text(j, i, conf_matrix_rf[i, j],
                 ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Per class performance - Random Forest
print("\n" + "="*60)
print("PER CLASS PERFORMANCE - RANDOM FOREST")
print("="*60)

per_class_accuracy_rf = conf_matrix_rf.diagonal() / conf_matrix_rf.sum(axis=1)

for i, class_label in enumerate(wine.target_names):
    print(f"\nClass {class_label}")
    print(f"Accuracy: {per_class_accuracy_rf[i]:.4f}")
    print(f"Correct: {conf_matrix_rf[i, i]}")
    print(f"Total: {conf_matrix_rf.sum(axis=1)[i]}")

# Confusion Matrix Visualization - SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_svm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix - SVM")
plt.colorbar()

plt.xticks(range(len(wine.target_names)), wine.target_names, rotation=45)
plt.yticks(range(len(wine.target_names)), wine.target_names)

# Add text annotations
for i in range(conf_matrix_svm.shape[0]):
    for j in range(conf_matrix_svm.shape[1]):
        plt.text(j, i, conf_matrix_svm[i, j],
                 ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Per class performance - SVM
print("\n" + "="*60)
print("PER CLASS PERFORMANCE - SVM")
print("="*60)

per_class_accuracy_svm = conf_matrix_svm.diagonal() / conf_matrix_svm.sum(axis=1)

for i, class_label in enumerate(wine.target_names):
    print(f"\nClass {class_label}")
    print(f"Accuracy: {per_class_accuracy_svm[i]:.4f}")
    print(f"Correct: {conf_matrix_svm[i, i]}")
    print(f"Total: {conf_matrix_svm.sum(axis=1)[i]}")

# Average confidence - Random Forest
avg_confidence_rf = np.mean(np.max(y_pred_proba_rf, axis=1))
print("\n" + "="*60)
print("AVERAGE CONFIDENCE - RANDOM FOREST")
print("="*60)
print(f"Average confidence: {avg_confidence_rf:.4f}")

# Average confidence - SVM
avg_confidence_svm = np.mean(np.max(y_pred_proba_svm, axis=1))
print("\n" + "="*60)
print("AVERAGE CONFIDENCE - SVM")
print("="*60)
print(f"Average confidence: {avg_confidence_svm:.4f}")

# Cross validation - Random Forest
from sklearn.model_selection import cross_val_score

print("\n" + "="*60)
print("CROSS VALIDATION - RANDOM FOREST")
print("="*60)

cv_scores_rf = cross_val_score(rf_clf, X_train, y_train, cv=5)
print(f"Scores: {cv_scores_rf}")
print(f"Mean: {cv_scores_rf.mean():.4f}")
print(f"Std: {cv_scores_rf.std():.4f}")

# Cross validation - SVM
print("\n" + "="*60)
print("CROSS VALIDATION - SVM")
print("="*60)

cv_scores_svm = cross_val_score(svm_clf, X_train, y_train, cv=5)
print(f"Scores: {cv_scores_svm}")
print(f"Mean: {cv_scores_svm.mean():.4f}")
print(f"Std: {cv_scores_svm.std():.4f}")

# 8. Comparison and Discussion
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
    print(f"Even after tuning, Random Forest (accuracy: {rf_accuracy:.4f}) still performs better")
