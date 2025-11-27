import pandas as pd

df = pd.read_csv("train.csv")

# Parse date
df["date"] = pd.to_datetime(df["date"])

# Basic time features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek  # 0 = Monday

# Use onpromotion as‑is
daily = df.groupby("date", as_index=False).agg(
    sales=("sales", "sum"),
    onpromotion=("onpromotion", "sum")
)

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day"] = daily["date"].dt.day
daily["dayofweek"] = daily["date"].dt.dayofweek

from sklearn.model_selection import train_test_split

feature_cols = ["year", "month", "day", "dayofweek", "onpromotion"]
X = daily[feature_cols]
y = daily["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(feature_cols, model.coef_)))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R^2:", r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual sales")
plt.ylabel("Predicted sales")
plt.title("Linear Regression: Actual vs Predicted Sales")
plt.tight_layout()
plt.show()
