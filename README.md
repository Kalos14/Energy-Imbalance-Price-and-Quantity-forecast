# Energy-Imbalance-Price-and-Quantity-forecast

Forecasting pipeline for 15-minute **imbalance prices** and **system state probabilities** on the Italian (or similar) power market.

At each forecast origin \((D, t)\) the models produce 24-step-ahead forecasts:

- \( \hat P^{\text{long}}_{D,t}(h) \) – conditional price if the system is Long  
- \( \hat P^{\text{short}}_{D,t}(h) \) – conditional price if the system is Short  
- \( \hat\pi_{D,t}(h) = \mathbb P(Y_{D,t+h} = \text{Long}) \) – probability of the system being Long  

for horizons \(h = 1, \dots, 24\).

The repository was developed as part of a technical assignment for a short-term power trading internship.
Runtimes(displyed on temrinal) : Linear Benchmark 1min 08s, Ridge 1min 46s, XGBoost 9min 32s    
---

## 1. Project Structure

```text
.
├── Datasets/
│   ├── imb_price.parquet        # Imbalance prices (EUR/MWh)
│   └── imb_quantity.parquet     # Imbalance quantities (MWh)
├── Main.py                      # End-to-end training & evaluation pipeline
├── Functions_file.py            # Feature engineering & modelling utilities
├── Dockerfile                   # Docker setup (CPU only)
├── requirements.txt             # Python dependencies
└── output/                      # Created at runtime (models, metrics, plots)


