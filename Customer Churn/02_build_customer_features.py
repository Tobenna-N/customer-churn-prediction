import pandas as pd
import os


BASE_PATH = r"C:\Users\offic\OneDrive\Desktop\LEARNING\Customer Churn"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")

prior_full = pd.read_csv(os.path.join(OUTPUT_PATH, "prior_full.csv"))
orders_prior = pd.read_csv(os.path.join(OUTPUT_PATH, "orders_prior.csv"))

#Customer order features
customer_orders = orders_prior.groupby("user_id").agg(
    total_orders=("order_number", "max"),
    avg_days_between_orders=("days_since_prior_order", "mean"),
    std_days_between_orders=("days_since_prior_order", "std"),
    avg_order_dow=("order_dow", "mean"),
    avg_order_hour=("order_hour_of_day", "mean")
).reset_index()

#Basket size features
basket_size_per_order = prior_full.groupby("order_id").size().reset_index(name="basket_size")

basket_size_with_user = basket_size_per_order.merge(
    orders_prior[["order_id", "user_id"]],
    on="order_id",
    how="left"
)

customer_basket = basket_size_with_user.groupby("user_id").agg(
    avg_basket_size=("basket_size", "mean"),
    max_basket_size=("basket_size", "max"),
    min_basket_size=("basket_size", "min")
).reset_index()

#product diversity features
customer_products = prior_full.groupby("user_id").agg(
    total_products_bought=("product_id", "count"),
    unique_products_bought=("product_id", "nunique"),
    unique_aisles_bought=("aisle_id", "nunique"),
    unique_departments_bought=("department_id", "nunique")
).reset_index()

#Reorder features
customer_reorder = prior_full.groupby("user_id").agg(
    total_reordered_items=("reordered", "sum"),
    reorder_rate=("reordered", "mean")
).reset_index()

#Cart position features
customer_cart_order = prior_full.groupby("user_id").agg(
    avg_add_to_cart_order=("add_to_cart_order", "mean")
).reset_index()

# Merge all customer features
customers = customer_orders.merge(customer_basket, on="user_id", how="left")
customers = customers.merge(customer_products, on="user_id", how="left")
customers = customers.merge(customer_reorder, on="user_id", how="left")
customers = customers.merge(customer_cart_order, on="user_id", how="left")

# handle missing values
customers["std_days_between_orders"] = customers["std_days_between_orders"].fillna(0)

print("Customer features shape:", customers.shape)
print(customers.head())
print(customers.isnull().sum())

customers.to_csv(os.path.join(OUTPUT_PATH, "customers_features_base.csv"), index=False)
print("Saved customers_features_base.csv")