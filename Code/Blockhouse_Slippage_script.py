import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv('CRWV//CRWV_2025-04-03 00_00_00+00_00.csv')

df['ts_event'] = pd.to_datetime(df['ts_event'])
df = df.sort_values('ts_event')
df['minute'] = df['ts_event'].dt.floor('T')  
minute_snapshots = df.groupby('minute').tail(1)  

order_sizes = np.arange(10, 600, 10)  
all_slippages = []
all_order_sizes = []
all_minutes = []

for idx, row in minute_snapshots.iterrows():
    ask_prices = [row[f'ask_px_{str(i).zfill(2)}'] for i in range(10)]
    ask_sizes = [row[f'ask_sz_{str(i).zfill(2)}'] for i in range(10)]
    mid_price = (row['bid_px_00'] + row['ask_px_00']) / 2

    for x in order_sizes:
        remaining = x
        total_cost = 0
        for price, size in zip(ask_prices, ask_sizes):
            take = min(remaining, size)
            total_cost += take * price
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            continue
        avg_price = total_cost / x
        slippage = avg_price - mid_price

        all_slippages.append(slippage)
        all_order_sizes.append(x)
        all_minutes.append(row['minute'])


slippage_df = pd.DataFrame({
    'minute': all_minutes,
    'order_size': all_order_sizes,
    'slippage': all_slippages
})


mean_slippage = slippage_df.groupby('order_size')['slippage'].mean().reset_index()
X = mean_slippage['order_size'].values.reshape(-1, 1)
y = mean_slippage['slippage'].values

# Linear Regression 
lin_model = LinearRegression()
lin_model.fit(X, y)
y_pred_linear = lin_model.predict(X)

# Polynomial features (degree 2: quadratic)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# MSE and RMSE errors
mse_linear = mean_squared_error(y, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
print(f"Linear Regression - MSE: {mse_linear:.6f}, RMSE: {rmse_linear:.6f}")

mse_poly = mean_squared_error(y, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
print(f"Polynomial Regression - MSE: {mse_poly:.6f}, RMSE: {rmse_poly:.6f}")

plt.figure(figsize=(10, 5))
plt.scatter(mean_slippage['order_size'], mean_slippage['slippage'], label='Mean Slippage per Order Size', color='blue')
plt.plot(mean_slippage['order_size'], y_pred_linear, color='red', label='Linear Regression')
plt.plot(X, y_pred_poly, color='green', label='Polynomial Regression (degree 2)')
plt.xlabel('Order Size')
plt.ylabel('Average Slippage ($)')
plt.title('Average Slippage vs Order Size (Linear vs Nonlinear Regression)')
plt.legend()
plt.show()

