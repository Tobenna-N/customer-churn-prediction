import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


BASE_PATH = r"C:\Users\offic\OneDrive\Desktop\LEARNING\Customer Churn"
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")

customers = pd.read_csv(os.path.join(OUTPUT_PATH, "customers_features.csv"))

#select clustering features 
cluster_features = customers[[
    "total_orders",
    "avg_days_between_orders",
    "avg_basket_size",
    "reorder_rate",
    "unique_products_bought"
]]

# scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

#kmeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
customers["segment"] = kmeans.fit_predict(scaled_features)

# segement summary
segment_summary = customers.groupby("segment")[[
    "total_orders",
    "avg_days_between_orders",
    "avg_basket_size",
    "reorder_rate",
    "unique_products_bought",
    "churn"
]].mean().round(2)

print(segment_summary)

customers.to_csv(os.path.join(OUTPUT_PATH, "customers_segmented.csv"), index=False)
segment_summary.to_csv(os.path.join(OUTPUT_PATH, "segment_summary.csv"))

print("Saved customers_segmented.csv")
print("Saved segment_summary.csv")