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
