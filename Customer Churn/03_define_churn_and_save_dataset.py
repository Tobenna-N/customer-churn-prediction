import pandas as pd
import os

BASE_PATH = r"C:\Users\offic\OneDrive\Desktop\LEARNING\Customer Churn"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")

customers = pd.read_csv(os.path.join(OUTPUT_PATH, "customers_features_base.csv"))

# define churn
# churn = 1 if customer has long average gaps and relatively few orders

customers["churn"] = (
    (customers["avg_days_between_orders"] > 20) &
    (customers["total_orders"] <= 10)
).astype(int)

print(customers["churn"].value_counts())
print(customers["churn"].value_counts(normalize=True))

customers.to_csv(os.path.join(OUTPUT_PATH, "customers_features.csv"), index=False)
print("Saved customers_features.csv")