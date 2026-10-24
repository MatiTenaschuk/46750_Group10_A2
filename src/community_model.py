import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.data_loader import load_community

def add_member(m, d, tag):
    """Add the variables of one member to the Gurobi model m.

    m   : Gurobi model
    d   : InputData of the member
    tag : name prefix for the variables (e.g. "C1")
    """
    T = range(d.n_hours)   # hours 0..23
    v = {}                 # dictionary with all the member's variables

    v["load"] = m.addVars(T, lb=0, name=f"{tag}_load")        # consumption l_t (kWh/h)
    v["dev_pos"] = m.addVars(T, lb=0, name=f"{tag}_dev_pos")  # deviation above the reference (kWh/h)
    v["dev_neg"] = m.addVars(T, lb=0, name=f"{tag}_dev_neg")  # deviation below the reference (kWh/h)
    v["pv"] = m.addVars(T, lb=0, name=f"{tag}_pv")            # PV production used g_t (kWh/h)


    c = {}   # dictionary with all the member's constraints

    # Load = reference + deviation above - deviation below
    c["deviation"] = m.addConstrs(
        (v["load"][t] == d.reference_load[t] + v["dev_pos"][t] - v["dev_neg"][t] for t in T),
        name=f"{tag}_deviation")

    # Hourly bounds: reference +- 1 kWh (within 0..3 kWh/h)
    c["load_lower"] = m.addConstrs(
        (v["load"][t] >= d.load_lower_kWh[t] for t in T), name=f"{tag}_load_lower")
    c["load_upper"] = m.addConstrs(
        (v["load"][t] <= d.load_upper_kWh[t] for t in T), name=f"{tag}_load_upper")

    # Minimum daily energy: total consumption >= daily reference energy
    c["min_daily_energy"] = m.addConstr(
        gp.quicksum(v["load"][t] for t in T) >= d.min_daily_energy_kWh,
        name=f"{tag}_min_daily_energy")

    # PV: cannot use more than what is available (0 for members without PV)
    c["pv_limit"] = m.addConstrs(
        (v["pv"][t] <= d.pv_available[t] for t in T), name=f"{tag}_pv_limit")


    # Battery (only members that have one)
    if d.has_battery:
        v["charge"] = m.addVars(T, lb=0, ub=d.battery_max_charge_kW, name=f"{tag}_charge")          # p_ch (kW)
        v["discharge"] = m.addVars(T, lb=0, ub=d.battery_max_discharge_kW, name=f"{tag}_discharge") # p_dis (kW)
        v["soc"] = m.addVars(T, lb=0, ub=d.battery_capacity_kWh, name=f"{tag}_soc")                 # state of charge (kWh)

        eta_ch = d.battery_charging_efficiency
        eta_dis = d.battery_discharging_efficiency

        # State of charge: previous SoC + charged energy - discharged energy (with losses)
        c["soc_dynamics"] = m.addConstrs(
            (v["soc"][t] == (d.battery_initial_soc_kWh if t == 0 else v["soc"][t - 1])
             + eta_ch * v["charge"][t] - v["discharge"][t] / eta_dis for t in T),
            name=f"{tag}_soc_dynamics")

        # End of day: at least the initial state of charge (same choice as Assignment 1)
        c["soc_end"] = m.addConstr(
            v["soc"][T[-1]] >= d.battery_initial_soc_kWh, name=f"{tag}_soc_end")

    # Net consumption in every hour: what the member needs from the grid (negative = surplus)
    net = [v["load"][t] - v["pv"][t] for t in T]
    if d.has_battery:
        net = [net[t] + v["charge"][t] - v["discharge"][t] for t in T]

    # Own cost: PV marginal cost + disutility of deviating from the reference (DKK)
    own_cost = gp.quicksum(
        d.pv_marginal_cost * v["pv"][t]
        + d.linear_disutility * (v["dev_pos"][t] + v["dev_neg"][t])
        for t in T)
    
    return v, c, net, own_cost

def solve_reference(com):
    """Model R: every member alone, no contract, full tariffs.

    Solves one LP per member and returns a dict {consumer_id: results},
    where results holds the hourly import/export (kWh/h) and the total cost (DKK).
    """
    results = {}

    for d in com.members:
        T = range(d.n_hours)
        m = gp.Model(f"R_{d.consumer_id}")
        m.Params.OutputFlag = 0          # silence the Gurobi log

        v, c, net, own_cost = add_member(m, d, d.consumer_id)

        # Grid exchange on the member's own meter
        q_imp = m.addVars(T, lb=0, name="q_imp")
        q_exp = m.addVars(T, lb=0, name="q_exp")
        m.addConstrs((q_imp[t] - q_exp[t] == net[t] for t in T), name="balance")

        # Objective: full tariffs on import and export, plus own cost
        p, tau_imp, tau_exp = d.energy_price, d.import_tariff, d.export_tariff
        m.setObjective(
            gp.quicksum((p[t] + tau_imp) * q_imp[t] - (p[t] - tau_exp) * q_exp[t] for t in T) + own_cost,
            GRB.MINIMIZE)

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model R, member {d.consumer_id}: not optimal (status {m.Status})")

        results[d.consumer_id] = {
            "imp": np.array([q_imp[t].X for t in T]),
            "exp": np.array([q_exp[t].X for t in T]),
            "cost": m.ObjVal,
        }

    return results


def solve_individual(com):
    """Model I: every member signs its own contract (individual cap, discounted tariffs, penalty).

    Solves one LP per member and returns a dict {consumer_id: results},
    with the hourly import/export (kWh/h), the import above the own cap (kWh/h) and the total cost (DKK).
    Choice: kWh above the cap pay price + penalty, no tariff (assignment text).
    """
    beta = com.contract.tariff_discount_ratio            # 0.75
    alpha = com.contract.penalty_DKK_per_kWh             # 10 DKK/kWh
    results = {}

    for d in com.members:
        T = range(d.n_hours)
        cap_i = com.contract.individual_capacity_limit_kW[d.consumer_id]   # 24 values, kW

        m = gp.Model(f"I_{d.consumer_id}")
        m.Params.OutputFlag = 0

        v, c, net, own_cost = add_member(m, d, d.consumer_id)

        # Import split in two parts: within the own cap and above it
        q_below = m.addVars(T, lb=0, name="q_below")     # import within the cap (kWh/h)
        q_above = m.addVars(T, lb=0, name="q_above")     # import above the cap (kWh/h)
        q_exp = m.addVars(T, lb=0, name="q_exp")         # export (kWh/h)

        m.addConstrs((q_below[t] <= cap_i[t] for t in T), name="own_cap")
        m.addConstrs((q_below[t] + q_above[t] - q_exp[t] == net[t] for t in T), name="balance")

        # Objective: discounted tariffs below the cap and on export, penalty above the cap
        p, tau_imp, tau_exp = d.energy_price, d.import_tariff, d.export_tariff
        m.setObjective(
            gp.quicksum(
                (p[t] + (1 - beta) * tau_imp) * q_below[t]
                + (p[t] + alpha) * q_above[t]
                - (p[t] - (1 - beta) * tau_exp) * q_exp[t]
                for t in T) + own_cost,
            GRB.MINIMIZE)

        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model I, member {d.consumer_id}: not optimal (status {m.Status})")

        above = np.array([q_above[t].X for t in T])
        results[d.consumer_id] = {
            "imp": np.array([q_below[t].X for t in T]) + above,
            "exp": np.array([q_exp[t].X for t in T]),
            "above": above,
            "cost": m.ObjVal,
        }

    return results

def solve_community(com):
    """Model C: community contract under central dispatch (direct control).

    One LP with all members: one meter, one cap, discounted tariffs and penalty at the connection point,
    and an hourly internal balance behind the meter. Returns a dict with the community's hourly import,
    export and import above the cap (kWh/h), each member's net consumption (kWh/h), the total cost (DKK)
    and the dual of the internal balance (internal price, DKK/kWh).
    Choice: kWh above the cap pay price + penalty, no tariff (same as model I).
    """
    beta = com.contract.tariff_discount_ratio
    alpha = com.contract.penalty_DKK_per_kWh
    cap = com.contract.capacity_limit_kW
    p, tau_imp, tau_exp = com.energy_price, com.import_tariff, com.export_tariff
    T = range(len(p))

    m = gp.Model("C")
    m.Params.OutputFlag = 0

    # All members in the same model
    nets, own_costs = {}, 0
    for d in com.members:
        v, c, net, own_cost = add_member(m, d, d.consumer_id)
        nets[d.consumer_id] = net
        own_costs += own_cost

    # Exchange at the community's connection point
    P_below = m.addVars(T, lb=0, name="P_below")   # import within the cap (kWh/h)
    P_above = m.addVars(T, lb=0, name="P_above")   # import above the cap (kWh/h)
    P_exp = m.addVars(T, lb=0, name="P_exp")       # export (kWh/h)

    m.addConstrs((P_below[t] <= cap[t] for t in T), name="cap")

    # Internal balance: what the meter imports - exports = sum of the members' net consumption
    balance = m.addConstrs(
        (P_below[t] + P_above[t] - P_exp[t] == gp.quicksum(nets[i][t] for i in nets) for t in T),
        name="internal_balance")

    m.setObjective(
        gp.quicksum(
            (p[t] + (1 - beta) * tau_imp) * P_below[t]
            + (p[t] + alpha) * P_above[t]
            - (p[t] - (1 - beta) * tau_exp) * P_exp[t]
            for t in T) + own_costs,
        GRB.MINIMIZE)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Model C: not optimal (status {m.Status})")

    above = np.array([P_above[t].X for t in T])
    return {
        "imp": np.array([P_below[t].X for t in T]) + above,
        "exp": np.array([P_exp[t].X for t in T]),
        "above": above,
        "net": {i: np.array([nets[i][t].getValue() for t in T]) for i in nets},
        "cost": m.ObjVal,
        "price": np.array([balance[t].Pi for t in T]),   # internal price lambda_t (DKK/kWh)
    }


if __name__ == "__main__":
    com = load_community("base", day="d1")
    cap = com.contract.capacity_limit_kW

    res_R = solve_reference(com)
    res_I = solve_individual(com)
    res_C = solve_community(com)

    # Same structure for the three models: aggregated import and total cost
    summary = {
        "R": (sum(r["imp"] for r in res_R.values()), sum(r["cost"] for r in res_R.values())),
        "I": (sum(r["imp"] for r in res_I.values()), sum(r["cost"] for r in res_I.values())),
        "C": (res_C["imp"], res_C["cost"]),
    }

    for name, (P_imp, cost) in summary.items():
        print(f"--- Model {name} ---")
        print("  Daily import (kWh):", round(P_imp.sum(), 2))
        print("  Peak import (kW):", round(P_imp.max(), 2))
        print("  Total cost (DKK):", round(cost, 2))
        print("  Exceedance of community cap (kWh):", round(np.maximum(P_imp - cap, 0).sum(), 2))
        print("  Hours above community cap:", int((P_imp > cap + 1e-6).sum()))

    print("Internal price C (DKK/kWh):", np.round(res_C["price"], 2))