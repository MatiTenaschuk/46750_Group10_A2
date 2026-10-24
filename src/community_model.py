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

if __name__ == "__main__":
    com = load_community("base", day="d1")
    res_R = solve_reference(com)

    # Aggregated import at the connection point (what the DSO sees)
    P_imp = sum(r["imp"] for r in res_R.values())
    cap = com.contract.capacity_limit_kW

    print("Daily import (kWh):", round(P_imp.sum(), 2))
    print("Peak import (kW):", round(P_imp.max(), 2))
    print("Total cost (DKK):", round(sum(r["cost"] for r in res_R.values()), 2))
    print("Exceedance (kWh):", round(np.maximum(P_imp - cap, 0).sum(), 2))
    print("Hours above cap:", int((P_imp > cap).sum()))
    print("Reference values:", com.contract.reference_model_R)