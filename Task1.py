import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# 1. Load dataset (adjust path if needed)
train_path = "train.csv"   # e.g. "../input/store-sales-time-series-forecasting/train.csv"
df = pd.read_csv(train_path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# 2. Parse date column and sort
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# 3. Check missing values
print("\nMissing values per column:")
print(df.isna().sum())

# 4. Descriptive statistics
print("\nNumeric summary:")
print(df.describe())

# 5. Target 'sales' statistics
print("\nSales mean:", df["sales"].mean())
print("Sales median:", df["sales"].median())
print("Sales mode:", df["sales"].mode().iloc[0])

# 6. Aggregate daily total sales over all stores and families
daily_sales = df.groupby("date", as_index=False)["sales"].sum()

# Line chart: daily sales over time
plt.figure(figsize=(14, 5))
plt.plot(daily_sales["date"], daily_sales["sales"])
plt.title("Total Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# 7. Monthly sales bar chart
monthly_sales = daily_sales.copy()
monthly_sales["year_month"] = monthly_sales["date"].dt.to_period("M")
monthly_sales = (
    monthly_sales.groupby("year_month", as_index=False)["sales"].sum()
)
monthly_sales["year_month"] = monthly_sales["year_month"].astype(str)

plt.figure(figsize=(16, 5))
sns.barplot(data=monthly_sales, x="year_month", y="sales", color="steelblue")
plt.xticks(rotation=90)
plt.title("Total Monthly Sales")
plt.xlabel("Year-Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
