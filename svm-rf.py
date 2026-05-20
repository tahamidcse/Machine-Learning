from sklearn.datasets import load_wine
wine = load_wine(as_frame=True)
print(wine.DESCR)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target,test_size=0.3, random_state=42)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
lin_clf=LinearSVC(max_iter=1_000_000, dual=True, random_state=42)
lin_clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
cross_val_score(lin_clf, X_train, y_train).mean()
lin_clf = make_pipeline(StandardScaler(),
                        LinearSVC(dual=True, random_state=42))
lin_clf.fit(X_train, y_train)
cross_val_score(lin_clf, X_train, y_train).mean()
svm_clf = make_pipeline(StandardScaler(), SVC(random_state=42))
svm_clf.fit(X_train, y_train)
cross_val_score(svm_clf, X_train, y_train).mean()
param_distrib = {
    "svc__gamma": loguniform(0.001, 0.1),
    "svc__C": uniform(1, 10)
}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distrib, n_iter=100, cv=5,
                                   random_state=42)
rnd_search_cv.fit(X_train, y_train)
rnd_search_cv.best_estimator_
rnd_search_cv.best_score_
rnd_search_cv.score(X_test, y_test)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ======================================================
# LOAD DATASET
# ======================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

column_names = [
    'Class','Alcohol','Malic acid','Ash',
    'Alcalinity of ash','Magnesium',
    'Total phenols','Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity','Hue',
    'OD280/OD315 of diluted wines',
    'Proline'
]

df = pd.read_csv(url, names=column_names)

print("Dataset shape:", df.shape)

X = df.drop("Class", axis=1)
y = df["Class"]

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ======================================================
# MODELS
# ======================================================

models = {

    "SVM":
    Pipeline([
        ('scaler', StandardScaler()),
        ('model',
         SVC(
             kernel='rbf',
             probability=True,
             random_state=42
         ))
    ]),

    "Random Forest":
    Pipeline([
        ('model',
         RandomForestClassifier(
             n_estimators=200,
             random_state=42
         ))
    ])
}

# ======================================================
# TRAIN + EVALUATE
# ======================================================

results=[]

for name, model in models.items():

    print("\n"+"="*60)
    print(name)
    print("="*60)

    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)

    cv_scores=cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    results.append({
        'Model':name,
        'Accuracy':accuracy,
        'CV Mean':cv_scores.mean(),
        'CV Std':cv_scores.std()
    })

    print(f"\nTest Accuracy: {accuracy:.4f}")

    print(
        f"Cross Validation Accuracy: "
        f"{cv_scores.mean():.4f}"
        f" (+/- {cv_scores.std():.4f})"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test,y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test,y_pred))

# ======================================================
# COMPARISON TABLE
# ======================================================

results_df=pd.DataFrame(results)

print("\n"+"="*70)
print("SVM vs RANDOM FOREST")
print("="*70)

print(results_df)

best_model=results_df.loc[
    results_df['Accuracy'].idxmax()
]

print("\nBest Model:")
print(best_model)

# ======================================================
# ACCURACY COMPARISON
# ======================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("SVM vs Random Forest Accuracy")

plt.tight_layout()
plt.show()

# ======================================================
# CROSS VALIDATION COMPARISON
# ======================================================

plt.figure(figsize=(8,5))

plt.errorbar(
    results_df["Model"],
    results_df["CV Mean"],
    yerr=results_df["CV Std"],
    marker='o'
)

plt.xlabel("Model")
plt.ylabel("Cross Validation Accuracy")
plt.title("Cross Validation Comparison")

plt.tight_layout()
plt.show()

# ======================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ======================================================

rf = models["Random Forest"]

feature_importance = (
    rf.named_steps['model']
    .feature_importances_
)

importance_df = pd.DataFrame({
    'Feature':X.columns,
    'Importance':feature_importance
})

importance_df=importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Features from Random Forest:")
print(importance_df.head(10))

plt.figure(figsize=(10,6))

plt.bar(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xticks(rotation=45)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()
plt.show()
