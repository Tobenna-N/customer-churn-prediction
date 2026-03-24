Instacart Customer Churn Analysis and Prediction
Overview

This project analyzes customer purchasing behavior using Instacart transaction data to identify the main drivers of customer churn and build models that predict churn risk. The analysis combines feature engineering, customer segmentation, and machine learning to generate actionable insights.

Dataset

The project uses the Instacart Market Basket Analysis dataset:

3.4 million orders
32 million product purchases
206,000 customers
49,000 products
Approach

The workflow includes:

Building customer-level behavioral features
Defining churn based on purchasing patterns
Performing exploratory data analysis
Segmenting customers using K-Means clustering
Training classification models (Logistic Regression and Random Forest)

Key Insights

Customers with longer gaps between orders are significantly more likely to churn
Low-frequency customers have the highest churn risk
High reorder rates are associated with strong customer loyalty

Results
Logistic Regression ROC-AUC: 0.997
Random Forest ROC-AUC: 1.00
Technologies Used

Python, pandas, numpy, matplotlib, seaborn, scikit-learn

See Report.pdf for detailed methodology, analysis, and business insights.
