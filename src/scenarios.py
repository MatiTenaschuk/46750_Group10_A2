"""Question 3: from the joint scenarios of scenario_generation.py to model data, and evaluation of posted prices.

    python -m src.scenarios          # draws the base in-sample and out-of-sample sets and runs the checks

Nothing here depends on how the pricing model of Question 2 is built: only on add_member() and CommunityModel
(Question 1). Three building blocks:

    draw_sets(com, n_in, n_out, seed_in, seed_out)   -> in-sample and out-of-sample joint scenarios (seeds reported)
    scenario_community(com, scenario)                -> copy of com with every member's data as the scenario sees it
    evaluate_prices(com, scenarios, x, tie)          -> what posted prices x cost in every scenario (one row each)

Notation: scenario s, probability pi_s, scaling parameter theta_{i,s} of member i (profile: rho_t ^ theta).
"""
import copy

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from src.data_loader import load_community
from src import scenario_generation as sg
from src.community_model import CommunityModel, add_member


# --- scenarios --------------------------------------------------------------------------------------------------
def draw_sets(com, n_in=10, n_out=100, seed_in=1, seed_out=2, spec=None):
    """In-sample and out-of-sample joint scenarios of the community, with the rules of the uncertainty file
    (or of `spec`, a specification tuned with sg.set_rule). Different seeds: the out-of-sample set is unseen.
    Nested in-sample sets: use nested(in_sample, n)."""
    app, par = com.raw["appliance_params"], com.raw["params"]      # day-1 catalogue: the base case (see day_data otherwise)
    rules = sg.load_json("preference_uncertainty_params.json")
    ins = sg.draw_preference_scenarios(app, par, rules, n=n_in, seed=seed_in, spec=copy.deepcopy(spec))
    oos = sg.draw_preference_scenarios(app, par, rules, n=n_out, seed=seed_out, spec=copy.deepcopy(spec))
    return ins, oos


def nested(scenarios, n):
    """The first n scenarios of a set, with probabilities 1/n (a smaller in-sample set inside a larger one)."""
    out = copy.deepcopy(scenarios[:n])
    for s in out:
        s["probability"] = 1.0 / n
    return out


def scenario_community(com, scenario):
    """Copy of the community in which every member has the data of one joint scenario:
        reference_load        = L^max x rho_t ^ theta           (kWh/h)
        min_daily_energy_kWh  = the scenario's daily reference energy (kWh)
        load_lower/upper_kWh  = reference -/+ max deviation, within [load_min, load_max]  (recomputed here: the
                                generator does not do it)
        linear_disutility     = c^L x theta_cL                  (unchanged in the base study)
    Prices, tariffs, contract, PV and battery are untouched."""
    new = copy.deepcopy(com)
    for d in new.members:
        load = next(a for a in scenario["members"][d.consumer_id].values() if "load_id" in a)
        ref = d.load_max_kWh * np.asarray(load["reference_load_capacity_factor"], dtype=float)
        dev = d.max_hourly_deviation_kWh
        d.reference_load = ref
        d.min_daily_energy_kWh = float(load["min_total_energy_per_day_kWh"])
        d.load_upper_kWh = np.minimum(d.load_max_kWh, ref + dev)
        d.load_lower_kWh = np.maximum(d.load_min_kWh, ref - dev)
        d.linear_disutility = float(load["linear_disutility_DKK_per_kWh"])
    return new


def benchmark_costs(com, scenarios, model="R"):
    """B_{i,s}: the cost of every member alone (model R, or I) in every scenario. DataFrame scenarios x members (DKK)."""
    rows = {}
    for s in scenarios:
        res = getattr(CommunityModel, model)(scenario_community(com, s)).solve()
        rows[s["id"]] = res["member_cost"]
    return pd.DataFrame(rows).T.rename_axis("scenario")


# --- evaluation of posted prices -----------------------------------------------------------------------------------
def _meter_bill(com, net_total):
    """What the manager pays the DSO for a given aggregated net consumption (24 values): import within the cap at the
    discounted tariff, import above it at the penalty, export at the discounted tariff. Same rule as model C."""
    c = com.contract
    p, cap = com.energy_price, c.capacity_limit_kW
    imp, exp = np.maximum(net_total, 0), np.maximum(-net_total, 0)
    below, above = np.minimum(imp, cap), np.maximum(imp - cap, 0)
    disc = 1 - c.tariff_discount_ratio
    bill = float((p + disc * com.import_tariff) @ below + (p + c.penalty_DKK_per_kWh) @ above
                 - (p - disc * com.export_tariff) @ exp)
    return bill, above


def respond(com_s, x, tie="optimistic", tol=1e-6):
    """The members' responses to the posted prices x (24 values, or {member: 24 values}) in one scenario.

    Step 1: every member minimizes its own cost (payment at x + PV cost + disutility): its optimal cost D_i.
    Step 2: its optimum is often not unique (linear disutility). Among the schedules that cost every member at most
            D_i + tol, take the one with the lowest ('optimistic') or highest ('pessimistic') community cost.
    Returns the net consumption, payment and own cost of every member, and the community's meter results."""
    T = range(len(com_s.energy_price))
    prices = x if isinstance(x, dict) else {i: x for i in com_s.member_ids}
    m = gp.Model("respond")
    m.Params.OutputFlag = 0
    net, own, pay = {}, {}, {}
    for d in com_s.members:
        i = d.consumer_id
        _, _, net[i], own[i] = add_member(m, d, i)
        pay[i] = gp.quicksum(prices[i][t] * net[i][t] for t in T)
    # step 1: the members are not coupled, so one LP gives every member's optimum
    m.setObjective(gp.quicksum(pay[i] + own[i] for i in net), GRB.MINIMIZE)
    m.optimize()
    best = {i: pay[i].getValue() + own[i].getValue() for i in net}
    # step 2: tie-break at the community level, every member staying optimal
    for i in net:
        m.addConstr(pay[i] + own[i] <= best[i] + tol)
    c = com_s.contract
    total = [gp.quicksum(net[i][t] for i in net) for t in T]
    if tie == "optimistic":
        # lowest community cost: the meter variables of model C on top of the members' optimal responses
        below = m.addVars(T, lb=0, ub=[float(v) for v in c.capacity_limit_kW])
        above, exp = m.addVars(T, lb=0), m.addVars(T, lb=0)
        m.addConstrs(below[t] + above[t] - exp[t] == total[t] for t in T)
        p, disc = com_s.energy_price, 1 - c.tariff_discount_ratio
        m.setObjective(gp.quicksum((p[t] + disc * com_s.import_tariff) * below[t] + (p[t] + c.penalty_DKK_per_kWh) * above[t]
                                   - (p[t] - disc * com_s.export_tariff) * exp[t] for t in T)
                       + gp.quicksum(own.values()), GRB.MINIMIZE)
    elif tie == "pessimistic":
        # largest import above the cap the members can cause while staying optimal: exc_t = max(total_t - cap_t, 0),
        # written with one binary per hour (maximizing a maximum is not a linear program)
        big = sum(d.load_max_kWh + (d.battery_max_charge_kW or 0.0) for d in com_s.members)
        exc, z = m.addVars(T, lb=0), m.addVars(T, vtype=GRB.BINARY)
        m.addConstrs(exc[t] <= total[t] - float(c.capacity_limit_kW[t]) + big * (1 - z[t]) for t in T)
        m.addConstrs(exc[t] <= big * z[t] for t in T)
        m.setObjective(gp.quicksum(exc[t] for t in T), GRB.MAXIMIZE)
    else:
        raise ValueError("tie must be 'optimistic' or 'pessimistic'")
    m.optimize()
    net_v = {i: np.array([net[i][t].getValue() for t in T]) for i in net}
    bill, above_v = _meter_bill(com_s, sum(net_v.values()))
    return {"net": net_v, "payment": {i: pay[i].getValue() for i in net}, "own_cost": {i: own[i].getValue() for i in net},
            "member_cost": best, "bill": bill, "above": above_v}


def evaluate_prices(com, scenarios, x, benchmark=None, tie="optimistic"):
    """Posted prices x on every scenario: the same evaluation for every model compared (deterministic, stochastic...).
    One row per scenario: probability, community cost and its parts, exceedance, manager's balance (collected - bill),
    and, with benchmark = benchmark_costs(...), the individual-rationality shortfall (sum and number of members)."""
    rows = []
    for s in scenarios:
        r = respond(scenario_community(com, s), x, tie)
        own, collected = sum(r["own_cost"].values()), sum(r["payment"].values())
        row = {"scenario": s["id"], "pi": s["probability"], "cost": r["bill"] + own, "bill": r["bill"], "own_cost": own,
               "exceedance_kWh": float(r["above"].sum()), "hours_above": int((r["above"] > 1e-6).sum()),
               "penalty": com.contract.penalty_DKK_per_kWh * float(r["above"].sum()),
               "manager_balance": collected - r["bill"]}
        if benchmark is not None:
            short = {i: max(0.0, r["member_cost"][i] - benchmark.loc[s["id"], i]) for i in r["member_cost"]}
            row["ir_shortfall"] = sum(short.values())
            row["members_worse_off"] = sum(v > 1e-6 for v in short.values())
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


if __name__ == "__main__":
    com = load_community("base", day="d1")
    ins, oos = draw_sets(com, n_in=10, n_out=100, seed_in=1, seed_out=2)
    print(f"in sample: {len(ins)} scenarios (seed 1), out of sample: {len(oos)} (seed 2)")

    # --- checks of the scenarios (assignment, Table 12: validation) ---------------------------------------------
    th = np.array([[s["theta"]["profile"][i] for i in com.member_ids] for s in oos])
    print(f"theta out of sample: min {th.min():.3f}, max {th.max():.3f}, mean {th.mean():.3f}  (bounds 0.8-1.2, mean 1)")
    cs = scenario_community(com, ins[0])
    gap = max(abs(d.min_daily_energy_kWh - d.reference_load.sum()) for d in cs.members)
    inside = all((d.load_lower_kWh <= d.reference_load + 1e-9).all() and (d.reference_load <= d.load_upper_kWh + 1e-9).all()
                 for d in cs.members)
    print(f"scenario 0: |E_min - reference energy| <= {gap:.1e} kWh; reference inside its hourly bounds: {inside}")
    print("nested: first 5 of the 10 in-sample draws equal a 5-scenario draw with the same seed:",
          all(abs(a["theta"]["profile"][i] - b["theta"]["profile"][i]) < 1e-12 for a, b in
              zip(nested(ins, 5), draw_sets(com, 5, 1, 1, 2)[0]) for i in com.member_ids))

    # --- limit case: theta = 1 and the internal prices of C must give back the cost of C ---------------------------
    res_C = CommunityModel.C(com).solve()
    base = [{"id": 0, "probability": 1.0, "members": sg.expected_preference_scenario(com.raw["appliance_params"], com.raw["params"])}]
    for tie in ("optimistic", "pessimistic"):
        ev = evaluate_prices(com, base, res_C["price"], tie=tie)
        print(f"theta = 1, prices of C, {tie:<11}: cost {ev['cost'][0]:8.2f} DKK (C: {res_C['cost']:.2f}), "
              f"exceedance {ev['exceedance_kWh'][0]:6.2f} kWh, manager keeps {ev['manager_balance'][0]:7.2f} DKK")

    # --- step 1 of Question 3: the deterministic prices facing the scenarios -----------------------------------------
    B = benchmark_costs(com, oos[:20])
    ev = evaluate_prices(com, oos[:20], res_C["price"], benchmark=B)
    print("\nprices of C on 20 out-of-sample scenarios (optimistic):")
    print(ev.describe().loc[["mean", "min", "max"]].round(2).to_string())