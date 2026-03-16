import pandas as pd
import os

#Load Data
BASE_PATH = r"C:\Users\offic\OneDrive\Desktop\LEARNING\Customer Churn"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

orders_path = os.path.join(BASE_PATH, "orders.csv")
prior_path = os.path.join(BASE_PATH, "order_products__prior.csv")
products_path = os.path.join(BASE_PATH, "products.csv")
aisles_path = os.path.join(BASE_PATH, "aisles.csv")
departments_path = os.path.join(BASE_PATH, "departments.csv")

orders = pd.read_csv(orders_path)
prior = pd.read_csv(prior_path)
products = pd.read_csv(products_path)
aisles = pd.read_csv(aisles_path)
departments = pd.read_csv(departments_path)

print("Orders:", orders.shape)
print("Prior:", prior.shape)
print("Products:", products.shape)
print("Aisles:", aisles.shape)
print("Departments:", departments.shape)

print(orders.head(), end="\n\n")
print(prior.head(), end="\n\n")
print(products.head(), end="\n\n")


print("\nMissing values in orders:")
print(orders.isnull().sum())

print("\nMissing values in prior:")
print(prior.isnull().sum())

print("\nMissing values in products:")
print(products.isnull().sum())


#Keep prior orders only
orders_prior = orders[orders["eval_set"] == "prior"].copy()
print("Orders prior:", orders_prior.shape)

#Merge product hierarchy
products_full = products.merge(aisles, on="aisle_id", how="left")
products_full = products_full.merge(departments, on="department_id", how="left")

#Merge all prior transactions
prior_full = prior.merge(
    orders_prior[[
        "order_id",
        "user_id",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order"
    ]],
    on="order_id",
    how="left"
)

prior_full = prior_full.merge(products_full, on="product_id", how="left")

print("Prepared prior_full shape:", prior_full.shape)
print(prior_full.head())

#save dataset
prior_full.to_csv(os.path.join(OUTPUT_PATH, "prior_full.csv"), index=False)
orders_prior.to_csv(os.path.join(OUTPUT_PATH, "orders_prior.csv"), index=False)

print("Saved:")
print("- prior_full.csv")
print("- orders_prior.csv")
