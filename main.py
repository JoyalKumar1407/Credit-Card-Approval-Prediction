import pandas as pd
import numpy as np
import pickle

# Uncomment after fixing matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Data Collection
# --------------------------------------------------

app = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

print("Application Dataset")
print(app.head())

print("\nCredit Dataset")
print(credit.head())

print("\nApplication Shape :", app.shape)
print("Credit Shape      :", credit.shape)

# --------------------------------------------------
# Univariate Analysis
# --------------------------------------------------

print("\nOccupation Type Count")
print(app["OCCUPATION_TYPE"].value_counts())

# Uncomment after fixing matplotlib

# sns.set(rc={'figure.figsize':(15,6)})
# sns.countplot(x="OCCUPATION_TYPE", data=app)
# plt.xticks(rotation=90)
# plt.show()

# --------------------------------------------------
# Descriptive Analysis
# --------------------------------------------------

print("\nDescriptive Statistics")
print(app.describe())

# --------------------------------------------------
# Data Pre-processing
# --------------------------------------------------

# Remove duplicate applicants
app.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values in Application Dataset")
print(app.isnull().sum())

# Fill missing values in OCCUPATION_TYPE
app["OCCUPATION_TYPE"] = app["OCCUPATION_TYPE"].fillna("Unknown")

# Convert credit status into binary target
# Good (1): C, X, 0
# Bad  (0): 1,2,3,4,5

credit["TARGET"] = credit["STATUS"].apply(
    lambda x: 1 if x in ["1", "2", "3", "4", "5"] else 0
)

credit = credit.groupby("ID")["TARGET"].max().reset_index()
# Each customer has multiple monthly records.
# Keep one final target per customer.
print("\nCredit Dataset After Grouping")
print(credit.head())

# Merge application and credit datasets
data = pd.merge(app, credit, on="ID", how="inner")

print("\nMerged Dataset Shape :", data.shape)

print("\nMerged Dataset Preview")
print(data.head())

print("\nMissing Values After Merge")
print(data.isnull().sum())
# --------------------------------------------------
# Feature Engineering & Handling Categorical Values
# --------------------------------------------------

# Display data types
print("\nData Types Before Encoding")
print(data.dtypes)

# List of categorical columns
categorical_columns = [
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE'
]

# Label Encoding
encoder = LabelEncoder()

for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

print("\nEncoded Dataset")
print(data.head())

print("\nData Types After Encoding")
print(data.dtypes)
print("\nTarget Distribution")
print(data["TARGET"].value_counts())

# --------------------------------------------------
# Model Building
# --------------------------------------------------

# Define Features (X) and Target (y)
X = data.drop(columns=["ID", "TARGET"])
y = data["TARGET"]

# Split the dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Dataset Shape :", X_train.shape)
print("Testing Dataset Shape  :", X_test.shape)

print("\nTraining Target Distribution")
print(y_train.value_counts())

print("\nTesting Target Distribution")
print(y_test.value_counts())

# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

# Create model
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
lr_model.fit(X_train, y_train)

# Predict
lr_pred = lr_model.predict(X_test)

# Evaluation
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("Accuracy :", round(lr_accuracy, 4))
print("F1 Score :", round(lr_f1, 4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

# Uncomment after fixing matplotlib
# plt.figure(figsize=(6,5))
# sns.heatmap(confusion_matrix(y_test, lr_pred),
#             annot=True,
#             fmt="d",
#             cmap="Blues")
# plt.title("Logistic Regression")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# --------------------------------------------------
# Decision Tree
# --------------------------------------------------

print("\n==============================")
print("DECISION TREE")
print("==============================")

# Create model
dt_model = DecisionTreeClassifier(
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Predict
dt_pred = dt_model.predict(X_test)

# Evaluation
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("Accuracy :", round(dt_accuracy, 4))
print("F1 Score :", round(dt_f1, 4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, dt_pred))

print("\nClassification Report")
print(classification_report(y_test, dt_pred))

# Uncomment after fixing matplotlib
# plt.figure(figsize=(6,5))
# sns.heatmap(confusion_matrix(y_test, dt_pred),
#             annot=True,
#             fmt="d",
#             cmap="Greens")
# plt.title("Decision Tree")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# --------------------------------------------------
# Random Forest
# --------------------------------------------------

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

# Create model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Accuracy :", round(rf_accuracy, 4))
print("F1 Score :", round(rf_f1, 4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# Uncomment after fixing matplotlib
# plt.figure(figsize=(6,5))
# sns.heatmap(confusion_matrix(y_test, rf_pred),
#             annot=True,
#             fmt="d",
#             cmap="Oranges")
# plt.title("Random Forest")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy
    ],
    "F1 Score": [
        lr_f1,
        dt_f1,
        rf_f1
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results)

# Best Model
best_model = rf_model

# Save Model
pickle.dump(best_model, open("model.pkl", "wb"))

print("\nBest Model Saved Successfully as model.pkl")