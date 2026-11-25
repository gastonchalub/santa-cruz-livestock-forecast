# Santa Cruz livestock forecast

Time-series analysis of sheep inventories in Santa Cruz, Argentina, 2008–2026, with exploratory forecasts through 2030.

## Historical data

19 annual observations recovered from the historical province table and annual sheets of the workbook named `000000_Evolución Existencias Ovinos por Categoria y Provincia serie 2008_2026.xlsx`. The original workbook is not included; the CSV was transcribed from its saved inspection output. The annual totals matched the historical table in that output. Original publisher and reference dates should be verified against the source workbook before formal reuse.

- 2008: 3,133,301 head
- 2010: 3,317,376 head
- 2026: 2,079,165 head

## Methodology

The reconstructed Python script compares last observation, damped Holt (phi=0.98), ARIMA(0,1,1), and log-linear trend using expanding training windows ending in 2018–2022 and four-year forecasts. MAE is calculated in head of livestock. It produces per-fold predictions, a model ranking, future forecasts and a plot.

## Earlier exploratory findings

The earlier project reported approximately 1.86 million head by 2030 (around 10.5% below 2026), with historical four-year error around 300,000 head. These figures are historical project notes, not a confirmed output of the reconstructed script. The error is not a confidence interval. The original model-selection code was not recovered.

## Run

```bash
pip install -r requirements.txt
python src/forecast.py
```

Outputs appear in `results/`. This reconstructed implementation has not been executed as part of publication; its scores may differ from the earlier experiment. Model selection on the same backtest can yield optimistic estimates. With only 19 observations and potential structural changes, forecasts are scenarios, not certainties. No causal inference is attempted.
