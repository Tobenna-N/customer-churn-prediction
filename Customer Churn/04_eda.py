import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

BASE_PATH = r"C:\Users\offic\OneDrive\Desktop\LEARNING\Customer Churn"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
PLOTS_PATH = os.path.join(OUTPUT_PATH, "plots")
os.makedirs(PLOTS_PATH, exist_ok=True)

customers = pd.read_csv(os.path.join(OUTPUT_PATH, "customers_features.csv"))

print(customers.head())
print(customers.describe())

#churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=customers, x="churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "01_churn_distribution.png"))
plt.show()

# total orders distributuin
plt.figure(figsize=(8, 5))
sns.histplot(customers["total_orders"], bins=50)
plt.title("Distribution of Total Orders")
plt.xlabel("Total Orders")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "02_total_orders_distribution.png"))
plt.show()

# average days betwenen orders
plt.figure(figsize=(8, 5))
sns.histplot(customers["avg_days_between_orders"], bins=40)
plt.title("Average Days Between Orders")
plt.xlabel("Average Days Between Orders")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "03_avg_days_between_orders.png"))
plt.show()

# average basket size 
plt.figure(figsize=(8, 5))
sns.histplot(customers["avg_basket_size"], bins=40)
plt.title("Average Basket Size Distribution")
plt.xlabel("Average Basket Size")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "04_avg_basket_size_distribution.png"))
plt.show()

# reorder rate vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=customers, x="churn", y="reorder_rate")
plt.title("Reorder Rate by Churn")
plt.xlabel("Churn")
plt.ylabel("Reorder Rate")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "05_reorder_rate_by_churn.png"))
plt.show()

#total orders vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=customers, x="churn", y="total_orders")
plt.title("Total Orders by Churn")
plt.xlabel("Churn")
plt.ylabel("Total Orders")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "06_total_orders_by_churn.png"))
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = customers.drop(columns=["user_id"]).corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "07_correlation_heatmap.png"))
plt.show()

print("EDA plots saved in outputs/plots/")