from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
data = pd.read_csv(ROOT / "data" / "sheep_stock.csv")
assert data.year.tolist() == list(range(2008, 2027))
assert data.sheep.gt(0).all()
y = data.sheep.to_numpy(dtype=float)

def predict(train, horizon, method):
    if method == "naive":
        return np.repeat(train[-1], horizon)
    if method == "holt_damped":
        fit = Holt(train / 1e6, damped_trend=True).fit(damping_trend=0.98)
        return np.asarray(fit.forecast(horizon)) * 1e6
    if method == "arima":
        return np.asarray(ARIMA(train / 1e6, order=(0,1,1)).fit().forecast(horizon)) * 1e6
    if method == "log_trend":
        x = np.arange(len(train))
        slope, intercept = np.polyfit(x, np.log(train), 1)
        return np.exp(intercept + slope * np.arange(len(train), len(train)+horizon))
    raise ValueError(method)

methods = ["naive", "holt_damped", "arima", "log_trend"]
rows = []
for cutoff in range(2018, 2023):
    n = cutoff - 2008 + 1
    for method in methods:
        forecasts = predict(y[:n], 4, method)
        for h, (actual, pred) in enumerate(zip(y[n:n+4], forecasts), 1):
            rows.append(dict(cutoff=cutoff, horizon=h, method=method, actual=actual, forecast=pred, absolute_error=abs(actual-pred)))
scores = pd.DataFrame(rows)
scores.to_csv(OUT / "backtest.csv", index=False)
summary = scores.groupby("method").absolute_error.mean().sort_values()
summary.to_csv(OUT / "mae.csv")
best = summary.index[0]
future = pd.DataFrame({"year": range(2027, 2031)})
for method in methods:
    future[method] = predict(y, 4, method)
future.to_csv(OUT / "forecast.csv", index=False)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data.year, y, marker="o", label="Observed")
ax.plot([2026, *future.year], [y[-1], *future[best]], marker="o", linestyle="--", label=best)
ax.set(xlabel="Year", ylabel="Sheep (head)", title="Santa Cruz sheep stock: historical series and forecast")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "forecast.png", dpi=160)
print(summary)
print(future)
