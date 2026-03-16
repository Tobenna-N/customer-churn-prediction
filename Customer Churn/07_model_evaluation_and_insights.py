import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")


BASE_PATH = r"C:\Users\offic\OneDrive\Desktop\LEARNING\Customer Churn"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
MODELS_PATH = os.path.join(OUTPUT_PATH, "models")
PLOTS_PATH = os.path.join(OUTPUT_PATH, "plots")
os.makedirs(PLOTS_PATH, exist_ok=True)

customers = pd.read_csv(os.path.join(OUTPUT_PATH, "customers_features.csv"))
results = pd.read_csv(os.path.join(MODELS_PATH, "model_predictions.csv"))

# evaluation
print("LOGISTIC REGRESSION REPORT")
print(classification_report(results["actual_churn"], results["logistic_pred"]))

print("RANDOM FOREST REPORT")
print(classification_report(results["actual_churn"], results["rf_pred"]))

print("Logistic ROC-AUC:", roc_auc_score(results["actual_churn"], results["logistic_prob"]))
print("Random Forest ROC-AUC:", roc_auc_score(results["actual_churn"], results["rf_prob"]))

# confusion matrix - Random forest 
cm = confusion_matrix(results["actual_churn"], results["rf_pred"])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "08_rf_confusion_matrix.png"))
plt.show()

# ROC curve
log_fpr, log_tpr, _ = roc_curve(results["actual_churn"], results["logistic_prob"])
rf_fpr, rf_tpr, _ = roc_curve(results["actual_churn"], results["rf_prob"])

plt.figure(figsize=(8, 6))
plt.plot(log_fpr, log_tpr, label="Logistic Regression")
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "09_roc_curve_comparison.png"))
plt.show()

# Random Forest Feature importance
X = customers.drop(columns=["user_id", "churn"])
y = customers["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x="importance", y="feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "10_feature_importance.png"))
plt.show()