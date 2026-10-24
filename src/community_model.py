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

class CommunityModel:
    """One parametrized model for the three setups of Question 1.

    Switches:
      meter        : "member"    -> each member has its own meter (tariffs, cap and penalty on its own flows)
                     "community" -> one meter for the whole community (tariffs, cap and penalty on the aggregate)
      use_cap      : True -> cap at each meter (individual caps if meter="member"), penalty above it
      use_discount : True -> discounted tariffs (1 - beta), False -> full tariffs

    Named configurations:
      R = member meter, no cap, no discount        (no contract)
      I = member meter, individual caps, discount  (individual contracts)
      C = community meter, community cap, discount (community contract, central dispatch)
    """

    def __init__(self, com, meter="community", use_cap=True, use_discount=True, name="C", verbose=False):
        if meter not in ("member", "community"):
            raise ValueError("meter must be 'member' or 'community'")
        self.com = com                      # CommunityData (members + contract)
        self.meter = meter
        self.use_cap = use_cap
        self.use_discount = use_discount
        self.name = name                    # label of the configuration ("R", "I", "C", ...)
        self.verbose = verbose              # True -> show the Gurobi log
        self.T = range(len(com.energy_price))
        self.ids = com.member_ids

    # --- Named configurations ------------------------------------------------------------------
    @classmethod
    def R(cls, com):
        """No contract: every member alone, full tariffs, no cap."""
        return cls(com, meter="member", use_cap=False, use_discount=False, name="R")

    @classmethod
    def I(cls, com):
        """Individual contracts: every member alone, own cap, discounted tariffs, penalty."""
        return cls(com, meter="member", use_cap=True, use_discount=True, name="I")

    @classmethod
    def C(cls, com):
        """Community contract under central dispatch: one meter, one cap, discounted tariffs, penalty."""
        return cls(com, meter="community", use_cap=True, use_discount=True, name="C")

    # --- Model building ------------------------------------------------------------------------
    def build(self):
        """Create the Gurobi model and add all members (variables, constraints, net consumption, own cost)."""
        self.m = gp.Model(self.name)
        self.m.Params.OutputFlag = int(self.verbose)

        self.net, self.own_cost = {}, {}
        for d in self.com.members:
            v, c, net, own_cost = add_member(self.m, d, d.consumer_id)
            self.net[d.consumer_id] = net
            self.own_cost[d.consumer_id] = own_cost
        # --- Meters: one per member (R, I) or one for the community (C) --------------------------
        T, ids, contract = self.T, self.ids, self.com.contract
        if self.meter == "member":
            self.meters = ids
            cap = {i: contract.individual_capacity_limit_kW[i] for i in ids}
            flow = {i: self.net[i] for i in ids}
        else:
            self.meters = ["community"]
            cap = {"community": contract.capacity_limit_kW}
            flow = {"community": [gp.quicksum(self.net[i][t] for i in ids) for t in T]}

        # Exchange with the grid at every meter (kWh/h)
        self.q_below = self.m.addVars(self.meters, T, lb=0, name="q_below")   # import within the cap
        self.q_above = self.m.addVars(self.meters, T, lb=0, name="q_above")   # import above the cap (penalized)
        self.q_exp = self.m.addVars(self.meters, T, lb=0, name="q_exp")       # export

        # Cap: limit the import within the cap; without cap, nothing can be "above"
        if self.use_cap:
            self.m.addConstrs((self.q_below[k, t] <= cap[k][t] for k in self.meters for t in T), name="cap")
        else:
            self.m.addConstrs((self.q_above[k, t] == 0 for k in self.meters for t in T), name="no_cap")

        # Balance at every meter: import - export = flow behind the meter
        # (for C: the internal balance, whose dual is the internal price)
        self.balance = self.m.addConstrs(
            (self.q_below[k, t] + self.q_above[k, t] - self.q_exp[k, t] == flow[k][t]
             for k in self.meters for t in T),
            name="balance")
        # --- Objective: what every meter pays to the DSO + every member's own cost ----------------
        beta = contract.tariff_discount_ratio if self.use_discount else 0.0
        alpha = contract.penalty_DKK_per_kWh
        p, tau_imp, tau_exp = self.com.energy_price, self.com.import_tariff, self.com.export_tariff

        # Payment of each meter (DKK). Choice: kWh above the cap pay price + penalty, no tariff
        self.cost_meter = {
            k: gp.quicksum(
                (p[t] + (1 - beta) * tau_imp) * self.q_below[k, t]
                + (p[t] + alpha) * self.q_above[k, t]
                - (p[t] - (1 - beta) * tau_exp) * self.q_exp[k, t]
                for t in T)
            for k in self.meters}

        self.m.setObjective(
            gp.quicksum(self.cost_meter.values()) + gp.quicksum(self.own_cost.values()), GRB.MINIMIZE)


        return self


    # --- Solving and results ---------------------------------------------------------------------
    def solve(self):
        """Build (if needed) and solve the model. Returns a dict with the same keys for R, I and C:
          model, cost (DKK), P_imp / P_exp / above (aggregated, kWh/h), net (per member, kWh/h),
          own_cost (per member, DKK); member_cost (per member, DKK) when meter="member";
          price (internal price, DKK/kWh) when meter="community".
        """
        if not hasattr(self, "m"):
            self.build()
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model {self.name}: not optimal (status {self.m.Status})")

        T, ids = self.T, self.ids
        res = {
            "model": self.name,
            "cost": self.m.ObjVal,
            "P_imp": np.array([sum(self.q_below[k, t].X + self.q_above[k, t].X for k in self.meters) for t in T]),
            "P_exp": np.array([sum(self.q_exp[k, t].X for k in self.meters) for t in T]),
            "above": np.array([sum(self.q_above[k, t].X for k in self.meters) for t in T]),
            "net": {i: np.array([self.net[i][t].getValue() for t in T]) for i in ids},
            "own_cost": {i: self.own_cost[i].getValue() for i in ids},
        }
        if self.meter == "member":
            # Each member pays its own meter: total cost of each member
            res["member_cost"] = {i: self.cost_meter[i].getValue() + res["own_cost"][i] for i in ids}
        else:
            # Dual of the internal balance: the internal price of central dispatch
            res["price"] = np.array([self.balance["community", t].Pi for t in T])
        return res


if __name__ == "__main__":
    com = load_community("base", day="d1")
    cap = com.contract.capacity_limit_kW

    results = {f.__name__: f(com).solve() for f in (CommunityModel.R, CommunityModel.I, CommunityModel.C)}

    for name, res in results.items():
        P_imp = res["P_imp"]
        print(f"--- Model {name} ---")
        print("  Daily import (kWh):", round(P_imp.sum(), 2))
        print("  Peak import (kW):", round(P_imp.max(), 2))
        print("  Total cost (DKK):", round(res["cost"], 2))
        print("  Exceedance of community cap (kWh):", round(np.maximum(P_imp - cap, 0).sum(), 2))
        print("  Hours above community cap:", int((P_imp > cap + 1e-6).sum()))

    print("Internal price C (DKK/kWh):", np.round(results["C"]["price"], 2))

    # Validation: the direct-control model with member meters, no cap and no discount (= configuration R)
    # must reproduce the reference values of the contract file
    print("--- Validation ---")
    print("  Model R:", {"daily_import_kWh": round(results["R"]["P_imp"].sum(), 2),
                         "peak_import_kW": round(results["R"]["P_imp"].max(), 2),
                         "cost_DKK": round(results["R"]["cost"], 2)})
    print("  File   :", com.contract.reference_model_R)
    