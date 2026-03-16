import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


BASE_PATH = r"C:\Users\offic\OneDrive\Desktop\LEARNING\Customer Churn"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
MODELS_PATH = os.path.join(OUTPUT_PATH, "models")
os.makedirs(MODELS_PATH, exist_ok=True)

customers = pd.read_csv(os.path.join(OUTPUT_PATH, "customers_features.csv"))

#features and targets
X = customers.drop(columns=["user_id", "churn"])
y = customers["churn"]

# train the test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# logistic regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

log_preds = log_model.predict(X_test_scaled)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

# random forest predictions
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

#save predictions and test sets
results = X_test.copy()
results["actual_churn"] = y_test.values
results["logistic_pred"] = log_preds
results["logistic_prob"] = log_probs
results["rf_pred"] = rf_preds
results["rf_prob"] = rf_probs

results.to_csv(os.path.join(MODELS_PATH, "model_predictions.csv"), index=False)

X_train.to_csv(os.path.join(MODELS_PATH, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(MODELS_PATH, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(MODELS_PATH, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(MODELS_PATH, "y_test.csv"), index=False)

print("Saved model_predictions.csv")
print("Model training complete.")