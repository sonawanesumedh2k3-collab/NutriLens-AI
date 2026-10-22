# ============================================================
# Nutritional Health Score Predictor
# Model Training Script
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/packaged_foods_india_cleaned.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CONVERT NUTRITION VALUES TO PER 100g
# ============================================================

df["calories_per_100g"] = (
    df["calories_kcal"] / df["serving_size_g"]
) * 100

df["sugar_per_100g"] = (
    df["sugar_g"] / df["serving_size_g"]
) * 100

df["saturated_fat_per_100g"] = (
    df["saturated_fat_g"] / df["serving_size_g"]
) * 100

df["sodium_per_100g"] = (
    df["sodium_mg"] / df["serving_size_g"]
) * 100

df["proteins_per_100g"] = (
    df["proteins_g"] / df["serving_size_g"]
) * 100

df["dietary_fiber_per_100g"] = (
    df["dietary_fiber_g"] / df["serving_size_g"]
) * 100


# ============================================================
# 3. SCORING FUNCTIONS
# ============================================================

def calorie_points(calories):

    if calories <= 80:
        return 0
    elif calories <= 160:
        return 1
    elif calories <= 240:
        return 2
    elif calories <= 320:
        return 3
    elif calories <= 400:
        return 4
    elif calories <= 480:
        return 5
    elif calories <= 560:
        return 6
    elif calories <= 640:
        return 7
    elif calories <= 720:
        return 8
    elif calories <= 800:
        return 9
    else:
        return 10


def sugar_points(sugar):

    if sugar <= 4.5:
        return 0
    elif sugar <= 9:
        return 1
    elif sugar <= 13.5:
        return 2
    elif sugar <= 18:
        return 3
    elif sugar <= 22.5:
        return 4
    elif sugar <= 27:
        return 5
    elif sugar <= 31.5:
        return 6
    elif sugar <= 36:
        return 7
    elif sugar <= 40.5:
        return 8
    elif sugar <= 45:
        return 9
    else:
        return 10


def saturated_fat_points(saturated_fat):

    if saturated_fat <= 1:
        return 0
    elif saturated_fat <= 2:
        return 1
    elif saturated_fat <= 3:
        return 2
    elif saturated_fat <= 4:
        return 3
    elif saturated_fat <= 5:
        return 4
    elif saturated_fat <= 6:
        return 5
    elif saturated_fat <= 7:
        return 6
    elif saturated_fat <= 8:
        return 7
    elif saturated_fat <= 9:
        return 8
    elif saturated_fat <= 10:
        return 9
    else:
        return 10


def sodium_points(sodium):

    if sodium <= 90:
        return 0
    elif sodium <= 180:
        return 1
    elif sodium <= 270:
        return 2
    elif sodium <= 360:
        return 3
    elif sodium <= 450:
        return 4
    elif sodium <= 540:
        return 5
    elif sodium <= 630:
        return 6
    elif sodium <= 720:
        return 7
    elif sodium <= 810:
        return 8
    elif sodium <= 900:
        return 9
    else:
        return 10


def protein_points(protein):

    if protein <= 1.6:
        return 0
    elif protein <= 3.2:
        return 1
    elif protein <= 4.8:
        return 2
    elif protein <= 6.4:
        return 3
    elif protein <= 8.0:
        return 4
    else:
        return 5


def fiber_points(fiber):

    if fiber <= 0.9:
        return 0
    elif fiber <= 1.8:
        return 1
    elif fiber <= 2.7:
        return 2
    elif fiber <= 3.6:
        return 3
    elif fiber <= 4.5:
        return 4
    else:
        return 5


# ============================================================
# 4. CALCULATE HEALTH SCORE
# ============================================================

df["health_score"] = (
    df["calories_per_100g"].apply(calorie_points)
    + df["sugar_per_100g"].apply(sugar_points)
    + df["saturated_fat_per_100g"].apply(saturated_fat_points)
    + df["sodium_per_100g"].apply(sodium_points)
    - df["proteins_per_100g"].apply(protein_points)
    - df["dietary_fiber_per_100g"].apply(fiber_points)
)


# ============================================================
# 5. CREATE HEALTH CATEGORY
# ============================================================

def health_category(score):

    if score <= 5:
        return "Healthy"

    elif score <= 18:
        return "Moderate"

    else:
        return "Unhealthy"


df["health_category"] = df["health_score"].apply(health_category)


print("\nHealth category distribution:")
print(df["health_category"].value_counts())


# ============================================================
# 6. SELECT FEATURES
# ============================================================

features = [
    "calories_per_100g",
    "sugar_per_100g",
    "saturated_fat_per_100g",
    "sodium_per_100g",
    "proteins_per_100g",
    "dietary_fiber_per_100g"
]

target = "health_category"


X = df[features]
y = df[target]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_predictions = logistic_model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


# ============================================================
# 10. DECISION TREE
# ============================================================

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(
    X_train_scaled,
    y_train
)

tree_predictions = tree_model.predict(X_test_scaled)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train_scaled,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test_scaled
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    f"Logistic Regression Accuracy : "
    f"{logistic_accuracy * 100:.2f}%"
)

print(
    f"Decision Tree Accuracy       : "
    f"{tree_accuracy * 100:.2f}%"
)

print(
    f"Random Forest Accuracy       : "
    f"{random_forest_accuracy * 100:.2f}%"
)


# ============================================================
# 13. RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        random_forest_predictions
    )
)


# ============================================================
# 14. RANDOM FOREST CONFUSION MATRIX
# ============================================================

print("========================================")
print("RANDOM FOREST CONFUSION MATRIX")
print("========================================")

print(
    confusion_matrix(
        y_test,
        random_forest_predictions
    )
)


# ============================================================
# 15. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("model", exist_ok=True)


# ============================================================
# 16. SAVE ALL MODELS
# ============================================================

joblib.dump(
    logistic_model,
    "model/logistic_regression_model.pkl"
)

joblib.dump(
    tree_model,
    "model/decision_tree_model.pkl"
)

joblib.dump(
    random_forest_model,
    "model/random_forest_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)


# ============================================================
# 17. SAVE THE BEST MODEL
# ============================================================

model_accuracies = {
    "Logistic Regression": logistic_accuracy,
    "Decision Tree": tree_accuracy,
    "Random Forest": random_forest_accuracy
}

best_model_name = max(
    model_accuracies,
    key=model_accuracies.get
)

print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Best model:", best_model_name)
print(
    f"Accuracy: "
    f"{model_accuracies[best_model_name] * 100:.2f}%"
)


# Save the best model separately
if best_model_name == "Logistic Regression":

    joblib.dump(
        logistic_model,
        "model/best_model.pkl"
    )

elif best_model_name == "Decision Tree":

    joblib.dump(
        tree_model,
        "model/best_model.pkl"
    )

else:

    joblib.dump(
        random_forest_model,
        "model/best_model.pkl"
    )


print("\n========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")

print("Created:")
print("model/logistic_regression_model.pkl")
print("model/decision_tree_model.pkl")
print("model/random_forest_model.pkl")
print("model/scaler.pkl")
print("model/best_model.pkl")