"""Question 1: sensitivity of the contract value to the contract terms (discount and cap).

    python -m src.sensitivity          # runs both sweeps, saves results/q1/sweep_*.csv and the figure
"""
import copy

import numpy as np
import pandas as pd

from src.data_loader import load_community
from src.community_model import CommunityModel, compute_metrics


# --- Variations of the base case (built in code, data files never edited) -----------------------
def with_discount(com, beta):
    """Copy of the community with the tariff discount ratio set to beta (0 = no discount, 1 = no tariff)."""
    new = copy.deepcopy(com)
    new.contract.tariff_discount_ratio = beta
    return new


def with_cap_factor(com, f):
    """Copy of the community with the cap scaled to energy factor f (base 1.3): cap = f x E_R / 24.

    The individual caps of model I are scaled by the same factor, so they still add up to the community cap.
    """
    new = copy.deepcopy(com)
    scale = f / com.contract.capacity_limit_daily_energy_factor
    new.contract.capacity_limit_kW = com.contract.capacity_limit_kW * scale
    new.contract.individual_capacity_limit_kW = {
        i: np.asarray(cap_i) * scale for i, cap_i in com.contract.individual_capacity_limit_kW.items()}
    return new


# --- Sweep -----------------------------------------------------------------------------------------
def run_sweep(com, variant, values, name):
    """Solve R, I and C for every value of one contract term and return the metrics as a table.

    com     : base-case CommunityData
    variant : function (com, value) -> modified copy of com (with_discount or with_cap_factor)
    values  : values of the contract term to sweep
    name    : name of the swept parameter (column of the table)
    Returns a DataFrame with one row per (value, model) and the metrics of compute_metrics (reference: R).
    """
    rows = []
    for value in values:
        com_v = variant(com, value)
        cap = com_v.contract.capacity_limit_kW
        results = {m: getattr(CommunityModel, m)(com_v).solve() for m in ("R", "I", "C")}
        for m, res in results.items():
            rows.append({name: value, "model": m, **compute_metrics(res, results["R"], cap)})
        print(f"  {name} = {value:.2f}: value of the contract I {rows[-2]['contract_value']:7.2f} DKK,"
              f" C {rows[-1]['contract_value']:7.2f} DKK")
    return pd.DataFrame(rows)


def break_even(df, name, model):
    """Value of the swept parameter at which the contract value of `model` crosses zero (linear interpolation).
    Returns None if it never crosses zero in the swept range."""
    d = df[df["model"] == model].sort_values(name)
    x, y = d[name].to_numpy(), d["contract_value"].to_numpy()
    for k in range(len(x) - 1):
        if y[k] == 0:
            return float(x[k])
        if y[k] * y[k + 1] < 0:
            return float(x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k]))
    return None


if __name__ == "__main__":
    from pathlib import Path
    from src.plotting import plot_sensitivity

    out = Path("results/q1")
    out.mkdir(parents=True, exist_ok=True)
    com = load_community("base", day="d1")

    print("Sweep of the discount beta (cap factor f = 1.3):")
    df_beta = run_sweep(com, with_discount, np.round(np.arange(0.0, 1.01, 0.1), 2), "beta")
    df_beta.to_csv(out / "sweep_beta.csv", index=False)

    print("Sweep of the cap factor f (discount beta = 0.75):")
    df_cap = run_sweep(com, with_cap_factor, np.round(np.arange(1.0, 2.01, 0.1), 2), "f")
    df_cap.to_csv(out / "sweep_cap.csv", index=False)

    for name, df in (("beta", df_beta), ("f", df_cap)):
        for m in ("I", "C"):
            print(f"Break-even {name} for model {m}: {break_even(df, name, m)}")

    plot_sensitivity(df_beta, df_cap, base_beta=0.75, base_f=1.3, path=str(out / "sensitivity"))
    print("Tables and figure saved in results/q1/")