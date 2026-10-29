# ⚡ GridCast FR

**Day-ahead forecast of French electricity consumption, with calibrated uncertainty and drift monitoring.**

Every night, GridCast downloads the national consumption published by RTE and the weather of the 10 largest French urban areas, then publishes the forecast of the next 24 hours with a 90 % prediction interval. A dashboard compares each forecast with reality, and automatic checks raise an alert when the model stops being trustworthy.

> 🔗 Live dashboard: _add your Streamlit link here_ · 🔗 API docs: _add your link here_

<!-- Add a screenshot of YOUR dashboard on real data: docs/dashboard.png -->

---

## Why this project

Grid operators and energy suppliers must know tomorrow's demand to plan production. A single number is not enough: a planner needs to know **how sure** the forecast is. This project answers three questions:

1. **How much** electricity will France consume tomorrow, hour by hour?
2. **How confident** are we? Every forecast comes with an interval that contains the real value 90 % of the time.
3. **Can we still trust the model today?** Accuracy, calibration and input drift are checked every day.

## Results

Rolling backtest over **52 weeks** (October 2025 – October 2026) on real RTE data.
Models learn with the observed temperature, but forecast with the **temperature forecast issued the day before**, as in production.

| Model | MAE (MW) | MAPE (%) | MAPE < 5 °C (%) | 90 % interval coverage | Mean width (MW) |
|---|---|---|---|---|---|
| Seasonal naive (same hour last week) | 3,508 | 6.55 | 12.26 | 90.4 % | 16,436 |
| Thermal physics model | 2,339 | 4.46 | 4.42 | 88.7 % | 8,855 |
| Hybrid (physics + LightGBM) | 1,284 | 2.50 | 3.05 | 88.3 % | 5,140 |
| **LightGBM (main model)** | **1,043** | **2.02** | 2.72 | 90.4 % | 4,495 |
| *RTE official day-ahead forecast (reference)* | 1,272 | 2.50 | 2.43 | – | – |

The last row is RTE's own forecast, published the day before on éCO2mix. It is used as a professional reference point, to put the error of a transparent and reproducible pipeline into perspective. GridCast forecasts are issued at midnight, so they benefit from a few more hours of observed consumption.

**Physical insight learned from the data:** each degree of cold below 15 °C adds about **2.3 GW** of demand in France, close to the order of magnitude published by RTE (about 2.4 GW per °C in winter).

### What the backtest taught me
- **Weather forecast error matters.** With the observed temperature, LightGBM reaches 1.71 % MAPE. With the temperature forecast of the day before, 2.02 %. Evaluating with observed weather would have given an optimistic result.
- **The hybrid model lost its advantage in realistic conditions.** With observed weather, it was the best model on cold hours. With forecast weather, its physics part passes the temperature error directly to the forecast, while LightGBM relies more on recent consumption. LightGBM was therefore chosen as the main model.
- **Intervals stay calibrated.** Coverage is 88 to 90 % for a 90 % target, and intervals widen automatically to include the weather forecast error.
- **Weak point: cold hours.** Below 5 °C, the error rises to 2.72 %. This is when the grid is under the most stress, and the first thing to improve.
- **Monitoring in production.** At the start of the heating season (October 2026), the monitoring raised an alert: interval coverage dropped to 78 % while accuracy stayed good (1.8 % MAPE). Intervals calibrated on the mild weeks before became too narrow when demand started reacting to cold.

## Architecture

```mermaid
flowchart TB
    A["RTE éCO2mix (ODRE API)<br/>consumption + RTE D-1 forecast"] --> C
    B["Open-Meteo<br/>temperature of 10 cities, population-weighted<br/>observed · forecast · forecast issued the day before"] --> C
    C[("DuckDB<br/>consumption · temperature · published forecasts")]

    C --> D["Features<br/>calendar · holidays · bridge days · degree-days · lags ≥ 24 h"]
    D --> E["4 models<br/>naive · physics · LightGBM · hybrid"]
    E --> F["Conformal prediction<br/>90 % interval per hour"]

    F --> BT["Rolling backtest<br/>with day-ahead weather, RTE as reference"]
    F --> FC["Next-day forecast"]
    FC --> MON["Monitoring<br/>accuracy · coverage · drift"]

    BT --> R[/"reports/"/]
    MON --> R
    FC --> S[/"artifacts/<br/>versioned model + metadata"/]
    FC -->|stored| C

    R --> SERVE["FastAPI · Streamlit dashboard"]
    C --> SERVE

    CRON(["GitHub Actions, every night"]) -.->|runs ingest → forecast → monitor| C
```

### What happens every night

```mermaid
sequenceDiagram
    autonumber
    participant GH as GitHub Actions (00:47 Paris)
    participant RTE as RTE / ODRE
    participant OM as Open-Meteo
    participant DB as DuckDB
    participant MOD as Models
    participant REP as reports/ + artifacts/

    GH->>RTE: download the last 10 days of consumption
    GH->>OM: download observed and forecast temperature
    RTE-->>DB: upsert consumption
    OM-->>DB: upsert temperature
    GH->>MOD: train on history, calibrate intervals
    MOD-->>DB: publish tomorrow's 24 hourly forecasts
    MOD-->>REP: save the model and its metadata
    GH->>REP: run monitoring checks (alert if drift)
    GH->>GH: commit the updated database and reports
```

### Models, from simple to hybrid

| Model | Idea | Why it is here |
|---|---|---|
| Seasonal naive | Same hour, one week before | The baseline every model must beat |
| Thermal physics | Linear model: daily profile + heating and cooling degree-days, with building inertia | Interpretable, every coefficient has a physical meaning |
| **LightGBM (main model)** | Gradient boosting on calendar, weather and lag features | Best model in realistic conditions, publishes the daily forecast |
| Hybrid | Physics model + LightGBM trained on its residuals | Physics gives the structure, ML corrects what physics misses |
| *RTE official D-1* | The forecast published by the grid operator | External reference, never used as an input |

### Honest evaluation
- **Rolling-origin backtest:** for each test week, models are trained only on the past, calibrated on the next 8 weeks they have never seen, then evaluated on the following week. Exactly what would have happened in production.
- **Day-ahead rule:** every feature is known at midnight before the target day (consumption lags ≥ 24 h). A unit test checks that no feature leaks the future.
- **Realistic weather:** calibration and test weeks use the temperature forecast issued the day before (Open-Meteo previous model runs), not the temperature observed afterwards. A unit test checks that a noisy forecast increases the error.
- **Professional reference:** RTE's official day-ahead forecast is scored on the same hours with the same metrics.
- **Robustness:** errors are also reported on cold hours (< 5 °C), when the grid is under stress.

### Calibrated uncertainty
Split conformal prediction, conditional on the hour of the day: the 90 % quantile of out-of-sample errors on the calibration window gives a margin with a coverage guarantee and no assumption on the error distribution. Night errors and evening-peak errors get different margins.

### Monitoring
| Check | Alert when |
|---|---|
| Rolling MAPE (14 days) | above 1.5 × the backtest MAPE |
| Interval coverage | below 80 % for a 90 % target |
| Temperature out of training range | more than 5 % of recent hours |
| Error distribution drift | Kolmogorov-Smirnov p-value < 0.01 |

### Data quality
`scripts/check_data.py` checks the data before any model: period covered, missing hours, consumption range, seasonality, temperature and daylight saving time. It revealed that the October clock change lost two hours per year: one is missing at the source, the other was lost by the pipeline because local time is ambiguous in the weather API. Weather data is now requested in UTC, so only the hour missing at the source remains (6 hours out of 51,000).

### Model registry
Every model used for a published forecast is saved in `artifacts/` (joblib) with a JSON file: training period, number of rows, target day and backtest MAPE. Any forecast can be traced back to the exact model that produced it, and reloaded to reproduce it.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
export PYTHONPATH=src:.                               # Windows PowerShell: $env:PYTHONPATH="src;."

# Real data
python -m gridcast.cli ingest --start 2021-01-01      # RTE + Open-Meteo
python scripts/check_data.py                          # data quality checks
python -m gridcast.cli ingest-forecast-weather --start 2025-01-01   # weather forecast issued the day before
python -m gridcast.cli backtest --weeks 52 --weather forecast       # realistic model comparison
python -m gridcast.cli replay --days 30               # past forecasts, to start monitoring
python -m gridcast.cli forecast                       # next day
python -m gridcast.cli monitor

# Dashboard and API
streamlit run app/dashboard.py
uvicorn api.main:app --reload                         # docs on http://localhost:8000/docs

# Tests
python -m pytest -q
```

No internet? `python -m gridcast.cli demo` loads synthetic data with the same structure, so the whole pipeline can run offline.

With Docker: `docker compose up --build` (API on port 8000, dashboard on 8501).

## Project structure

```
src/gridcast/
  data/rte.py        RTE eCO2mix download (ODRE API): consumption + RTE D-1 forecast, hourly resampling
  data/weather.py    Open-Meteo history, forecast and day-before forecast, population-weighted national temperature
  data/store.py      DuckDB storage (consumption, temperature, every published forecast)
  data/synthetic.py  synthetic data for tests and offline demo
  features.py        calendar, French holidays and bridge days, degree-days, lags
  models.py          naive, thermal physics, LightGBM, hybrid
  conformal.py       hourly split conformal prediction
  evaluate.py        rolling-origin backtest (observed or day-ahead weather), metrics, RTE reference
  monitoring.py      accuracy, coverage, out-of-range and drift checks
  registry.py        saves every trained model with its metadata
  pipeline.py        ingest, backtest, forecast, replay, monitor
  cli.py             command line
api/main.py          FastAPI service
app/dashboard.py     Streamlit dashboard
scripts/check_data.py  data quality checks
tests/               pytest: no future leakage, physics parameter recovery, conformal coverage,
                     realistic weather backtest, registry, API
.github/workflows/   CI tests on every push + nightly forecast job
reports/             backtest and monitoring results (created at run time)
artifacts/           versioned models (created at run time)
```

## Limitations and next steps
- Cold hours are the weak point (2.72 % MAPE below 5 °C): features such as the forecast of the coldest hour of the day, or a dedicated model for cold spells, could help.
- One year of test only: an exceptional cold wave is not in the test period.
- Temperature only: adding cloud cover (lighting) and wind chill would help.
- The physics model uses fixed degree-day thresholds (15 °C / 22 °C); they could be estimated from data.
- Adaptive conformal inference would react faster after a regime change.
- Probabilistic models (quantile regression) could be compared to conformal intervals.

## Data sources
- RTE, éCO2mix national data, via the ODRE open-data portal (Licence Ouverte / Etalab).
- Open-Meteo weather API: historical, forecast and previous model runs (CC BY 4.0).
