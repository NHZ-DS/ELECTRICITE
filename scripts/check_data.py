"""Quick data-quality check, to run after `ingest` and before any model.

    python scripts/check_data.py
"""
import pandas as pd

from gridcast.data.store import Store

store = Store(read_only=True)
try:
    df = store.load_dataset()
finally:
    store.close()

load = df.dropna(subset=["load_mw"])
print("1. Period covered")
print(f"   consumption from {load['ts'].min()} to {load['ts'].max()}")
print(f"   temperature until {df['ts'].max()}")

full = pd.date_range(load["ts"].min(), load["ts"].max(), freq="1h")
missing = full.difference(load["ts"])
print("\n2. Missing hours")
print(f"   {len(missing)} missing out of {len(full)} ({100 * len(missing) / len(full):.2f} %)")
if len(missing):
    days = pd.Series(missing.normalize()).value_counts().sort_index()
    print("   days with missing hours (first 10):")
    print(days.head(10).to_string())

print("\n3. Consumption range (expected roughly 30 000 to 90 000 MW)")
print(load["load_mw"].describe().round(0).to_string())

print("\n4. Winter higher than summer? Mean consumption by month (MW)")
print(load.groupby(load["ts"].dt.month)["load_mw"].mean().round(0).to_string())

print("\n5. Temperature")
print(f"   missing values: {df['temp_c'].isna().sum()}")
print(f"   min {df['temp_c'].min():.1f} °C, max {df['temp_c'].max():.1f} °C")

print("\n6. RTE official D-1 forecast")
share = 100 * load["rte_forecast_mw"].notna().mean()
print(f"   available on {share:.1f} % of the hours")
ok = load.dropna(subset=["rte_forecast_mw"])
if len(ok):
    mape = 100 * ((ok["load_mw"] - ok["rte_forecast_mw"]).abs() / ok["load_mw"]).mean()
    print(f"   RTE MAPE over the whole period: {mape:.2f} %")

print("\n7. Daylight saving time: number of hours on the change days")
for d in ["2025-03-30", "2025-10-26"]:
    n = (load["ts"].dt.strftime("%Y-%m-%d") == d).sum()
    print(f"   {d}: {n} hours (expected 23 in March, 25 in October)")