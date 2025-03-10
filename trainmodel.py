import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("Comprehensive_Blood_Report_Analysis.csv")  # Ensure correct path

# Encode categorical labels
label_encoders = {}
for col in ["Gender", "Diabetes", "Thyroid Disorder", "Liver Disease", "Kidney Disease", "UTI", "Typhoid", "Malaria", "High Cholesterol"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert categorical urine test values
df["Urine Protein"] = df["Urine Protein"].map({"Negative": 0, "Positive": 1})
df["Urine Glucose"] = df["Urine Glucose"].map({"Negative": 0, "Positive": 1})

# Select features and labels
X = df.drop(columns=["Patient_ID", "Diabetes", "Thyroid Disorder", "Liver Disease", "Kidney Disease", "UTI", "Typhoid", "Malaria", "High Cholesterol"])
y = df[["Diabetes", "Thyroid Disorder", "Liver Disease", "Kidney Disease", "UTI", "Typhoid", "Malaria", "High Cholesterol"]]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check class distribution
print("Disease Class Distribution Before Splitting:\n", y.sum(axis=0))

# Ensure each disease has at least 2 samples
for disease in y.columns:
    if y[disease].nunique() < 2:
        print(f"⚠ Warning: {disease} has only one class. Duplicating cases.")
        df = pd.concat([df, df[df[disease] == 1]], ignore_index=True)
        X = df.drop(columns=["Patient_ID", "Diabetes", "Thyroid Disorder", "Liver Disease", "Kidney Disease", "UTI", "Typhoid", "Malaria", "High Cholesterol"])
        y = df[["Diabetes", "Thyroid Disorder", "Liver Disease", "Kidney Disease", "UTI", "Typhoid", "Malaria", "High Cholesterol"]]
        X_scaled = scaler.fit_transform(X)

# Split data
try:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
except ValueError:
    print("⚠ Warning: Some diseases still have only one class. Stratify disabled.")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE for each disease separately
smote = SMOTE(sampling_strategy='auto', random_state=42)
balanced_X_train = []
balanced_y_train = []

for i, disease in enumerate(y.columns):
    X_res, y_res = smote.fit_resample(X_train, y_train[disease])
    balanced_X_train.append(X_res)
    balanced_y_train.append(y_res)

# Convert back to arrays
X_train = np.vstack(balanced_X_train)
y_train = np.column_stack(balanced_y_train)

# Train Random Forest model with improved hyperparameters
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

# Save feature names
joblib.dump(list(X.columns), "feature_names.pkl")  # ✅ Save feature names

# Save the trained model and scaler
joblib.dump(rf_model, "trained_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Predict and evaluate
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)  # Get probability scores for each class

print("\nPredicted Labels:\n", y_pred)
print("\nPrediction Probabilities:\n", y_proba)

accuracies = {disease: accuracy_score(y_test[disease], y_pred[:, i]) for i, disease in enumerate(y.columns)}
print("Model Training Complete! Accuracy per disease:")
for disease, acc in accuracies.items():
    print(f"{disease}: {acc * 100:.2f}%")

# Print detailed classification report
from collections import Counter
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=y.columns))

# Check if predictions are skewed toward one disease
label_counts = Counter(np.argmax(y_pred, axis=1))
print("\nPrediction Distribution:", label_counts)

# Print feature importance
feature_importances = rf_model.feature_importances_
sorted_features = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)
print("\nFeature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")
