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
