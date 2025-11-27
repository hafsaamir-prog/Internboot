import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare
df = pd.read_csv("train.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Aggregate to total daily sales (all stores, all families)
daily = df.groupby("date", as_index=False)["sales"].sum()
daily = daily.set_index("date")

# 7-day (weekly) simple moving average
daily["ma_7"] = daily["sales"].rolling(window=7).mean()

# 30-day (monthly) simple moving average
daily["ma_30"] = daily["sales"].rolling(window=30).mean()

# Shift MAs by 1 day so they only use past information
daily["forecast_ma_7"] = daily["ma_7"].shift(1)
daily["forecast_ma_30"] = daily["ma_30"].shift(1)

# Drop initial NaNs where we don't have enough history
eval_df = daily.dropna(subset=["forecast_ma_7", "forecast_ma_30"])

# Manual MAE
y_true = eval_df["sales"]
y_pred_7 = eval_df["forecast_ma_7"]
y_pred_30 = eval_df["forecast_ma_30"]

mae_7 = (y_true - y_pred_7).abs().mean()
mae_30 = (y_true - y_pred_30).abs().mean()

print("MAE (7-day MA forecast):", mae_7)
print("MAE (30-day MA forecast):", mae_30)

plt.figure(figsize=(14, 6))
plt.plot(daily.index, daily["sales"], label="Actual daily sales", alpha=0.4)
plt.plot(daily.index, daily["ma_7"], label="7-day moving average", linewidth=2)
plt.plot(daily.index, daily["ma_30"], label="30-day moving average", linewidth=2)
plt.title("Store Sales with Weekly and Monthly Moving Averages")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()
