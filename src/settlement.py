"""Question 1: settlement of the community contract with the internal price.

Every member pays the internal price lambda_t (dual of the internal balance of model C) for its net
consumption, and bears its own PV and disutility cost. The script checks

  * budget balance: what the manager collects from the members against what it pays at the meter;
  * individual rationality: every member's cost under C against its cost without contract (R) and
    under its own contract (I), by consumer type.

    python -m src.settlement          # prints the tables, saves results/q1/settlement.csv and the figure
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_community
from src.community_model import CommunityModel


ASSETS = {(False, False): "none", (True, False): "PV", (False, True): "battery", (True, True): "PV + battery"}


def settle(com, res_R, res_I, res_C):
    """Cost of every member under R, I and C, with C settled at the internal price.

    Returns a DataFrame with one row per member: its type (disutility slope, assets), its cost under R and
    under I (its own meter), its payment at the internal price and own cost under C, and its gain under C
    against R and against I (positive = better off in the community).
    """
    lam = res_C["price"]                                         # internal price, DKK/kWh
    rows = []
    for d in com.members:
        i = d.consumer_id
        payment = float(lam @ res_C["net"][i])                   # sum_t lambda_t * n_it
        cost_C = payment + res_C["own_cost"][i]
        rows.append({
            "member": i,
            "slope": d.linear_disutility,
            "assets": ASSETS[(d.has_pv, d.has_battery)],
            "cost_R": res_R["member_cost"][i],
            "cost_I": res_I["member_cost"][i],
            "payment_C": payment,
            "own_cost_C": res_C["own_cost"][i],
            "cost_C": cost_C,
        })
    df = pd.DataFrame(rows)
    df["gain_vs_R"] = df["cost_R"] - df["cost_C"]
    df["gain_vs_I"] = df["cost_I"] - df["cost_C"]
    return df


def budget_balance(res_C):
    """What the manager collects from the members at the internal price against what it pays at the meter.

    collected = sum_t lambda_t * (sum_i n_it) = sum_t lambda_t * (P_imp - P_exp)
    paid      = energy + grid tariffs + penalty at the connection point
    surplus   = collected - paid (> 0: the manager collects more than it pays, the congestion rent)
    """
    collected = float(res_C["price"] @ (res_C["P_imp"] - res_C["P_exp"]))
    paid = res_C["cost_energy"] + res_C["cost_tariffs"] + res_C["cost_penalty"]
    return {"collected": collected, "paid": paid, "surplus": collected - paid}


if __name__ == "__main__":
    from src.plotting import plot_settlement

    out = Path("results/q1")
    out.mkdir(parents=True, exist_ok=True)
    com = load_community("base", day="d1")
    res = {m: getattr(CommunityModel, m)(com).solve() for m in ("R", "I", "C")}

    df = settle(com, res["R"], res["I"], res["C"])
    bb = budget_balance(res["C"])
    df.to_csv(out / "settlement.csv", index=False)

    pd.set_option("display.width", 120)
    print(df.round(2).to_string(index=False))
    print("\n--- Budget balance of the manager (DKK) ---")
    print(f"  collected from the members : {bb['collected']:8.2f}")
    print(f"  paid at the meter          : {bb['paid']:8.2f}")
    print(f"  surplus                    : {bb['surplus']:8.2f}")
    print("  check: sum of member costs - cost of C =",
          round(df['cost_C'].sum() - res['C']['cost'], 2), "(must equal the surplus)")

    print("\n--- Individual rationality ---")
    print("  members worse off than without contract (R):", list(df.loc[df["gain_vs_R"] < -1e-6, "member"]))
    print("  members worse off than alone (I)           :", list(df.loc[df["gain_vs_I"] < -1e-6, "member"]))
    print("\n  Gain against R by slope and assets (DKK):")
    print(df.pivot(index="slope", columns="assets", values="gain_vs_R").round(2).to_string())

    plot_settlement(df, path=str(out / "settlement"))
    print("\nTable and figure saved in results/q1/")
