"""Question 1: community dispatch, contract sensitivity, and dual-price settlement."""

import copy
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np


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
        self.member_vars = {}                    # the variables for every member needed for the KKT conditions
        for d in self.com.members:
            v, c, net, own_cost = add_member(self.m, d, d.consumer_id)
            self.member_vars[d.consumer_id] = v
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

        self.beta, self.alpha = beta, alpha   # kept for the cost breakdown in solve()

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
        return self._collect()

    def _collect(self):
        """Read the solution into the result dict (separate from solve() so that Q2's PricingModel can reuse it)."""
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


        # Cost breakdown (DKK): energy at the day-ahead price, grid tariffs, cap penalty, own cost (PV + disutility)
        p, tau_imp, tau_exp = self.com.energy_price, self.com.import_tariff, self.com.export_tariff
        imp_below = np.array([sum(self.q_below[k, t].X for k in self.meters) for t in T])
        res["cost_energy"] = float(p @ (res["P_imp"] - res["P_exp"]))
        res["cost_tariffs"] = float((1 - self.beta) * (tau_imp * imp_below.sum() + tau_exp * res["P_exp"].sum()))
        res["cost_penalty"] = float(self.alpha * res["above"].sum())
        res["cost_own"] = float(sum(res["own_cost"].values()))


        if self.meter == "member":
            # Each member pays its own meter: total cost of each member
            res["member_cost"] = {i: self.cost_meter[i].getValue() + res["own_cost"][i] for i in ids}
        elif not self.m.IsMIP:
            # Dual of the internal balance: the internal price of central dispatch (an LP only, MILP has no duals)
            res["price"] = np.array([self.balance["community", t].Pi for t in T])
        return res

def compute_metrics(res, res_R, cap):
    """Metrics of one model against the reference model R (no contract).

    res   : result of CommunityModel.solve() for the model evaluated
    res_R : result of CommunityModel.R(com).solve() (the reference)
    cap   : community cap (kW, 24 values)

    Community point of view:
      contract_value      = cost_R - cost (DKK)            > 0: signing is worth it for the community
      cost_* components   (DKK)                            why the cost changes
    DSO point of view:
      exceedance          = sum_t max(P_imp - cap, 0)      congestion left (kWh)
      congestion_removed  = 1 - exceedance / exceedance_R  share of R's congestion removed (-)
      hours_above         = hours with P_imp > cap
      peak, peak_reduction = max P_imp (kW), 1 - peak / peak_R (-)
      tariff_revenue_lost = tariffs_R - tariffs (DKK)      what the discount costs the DSO
    """
    P, P_R = res["P_imp"], res_R["P_imp"]
    exceedance = float(np.maximum(P - cap, 0).sum())
    exceedance_R = float(np.maximum(P_R - cap, 0).sum())
    return {
        # Community
        "cost": res["cost"],
        "contract_value": res_R["cost"] - res["cost"],
        "cost_energy": res["cost_energy"],
        "cost_tariffs": res["cost_tariffs"],
        "cost_penalty": res["cost_penalty"],
        "cost_own": res["cost_own"],
        # DSO
        "exceedance": exceedance,
        "congestion_removed": 1 - exceedance / exceedance_R if exceedance_R > 0 else float("nan"),
        "hours_above": int((P > cap + 1e-6).sum()),
        "peak": float(P.max()),
        "peak_reduction": float(1 - P.max() / P_R.max()) if P_R.max() > 0 else float("nan"),
        "tariff_revenue_lost": res_R["cost_tariffs"] - res["cost_tariffs"],
    }


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
    import pandas as pd

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


def run_grid(com, betas, factors):
    """Value of the contract of I and C on a grid of discount (beta) x cap factor (f).

    The cost without contract (R) does not depend on the contract terms, so R is solved once.
    Returns a DataFrame with columns beta, f, model ("I" or "C") and contract_value = C^R - C^m (DKK).
    """
    import pandas as pd

    cost_R = CommunityModel.R(com).solve()["cost"]
    rows = []
    for f in factors:
        for beta in betas:
            com_v = with_discount(with_cap_factor(com, f), beta)
            for m in ("I", "C"):
                cost_m = getattr(CommunityModel, m)(com_v).solve()["cost"]
                rows.append({"beta": beta, "f": f, "model": m, "contract_value": cost_R - cost_m})
        print(f"  f = {f:.2f} done")
    return pd.DataFrame(rows)


ASSETS = {(False, False): "none", (True, False): "PV", (False, True): "battery", (True, True): "PV + battery"}


def settle(com, res_R, res_I, res_C):
    """Cost of every member under R, I and C, with C settled at the internal price.

    Returns a DataFrame with one row per member: its type (disutility slope, assets), its cost under R and
    under I (its own meter), its payment at the internal price and own cost under C, and its gain under C
    against R and against I (positive = better off in the community).
    """
    import pandas as pd

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


def run_sensitivity(com, out):
    """Write the discount and capacity sweeps and their combined grid."""
    from src.plotting import plot_sensitivity, plot_sensitivity_summary

    base_beta = com.contract.tariff_discount_ratio
    base_factor = com.contract.capacity_limit_daily_energy_factor
    betas = np.round(np.arange(0.0, 1.01, 0.1), 2)
    cap_factors = np.round(np.arange(1.0, 2.01, 0.1), 2)

    print(f"Sweep of the discount beta (cap factor f = {base_factor:g}):")
    df_beta = run_sweep(com, with_discount, betas, "beta")
    df_beta.to_csv(out / "sweep_beta.csv", index=False)

    print(f"Sweep of the cap factor f (discount beta = {base_beta:g}):")
    df_cap = run_sweep(com, with_cap_factor, cap_factors, "f")
    df_cap.to_csv(out / "sweep_cap.csv", index=False)

    for name, df in (("beta", df_beta), ("f", df_cap)):
        for m in ("I", "C"):
            print(f"Break-even {name} for model {m}: {break_even(df, name, m)}")

    plot_sensitivity(df_beta, df_cap, base_beta=base_beta, base_f=base_factor, path=str(out / "sensitivity"))

    print("Grid of discount x cap factor (value of the contract):")
    df_grid = run_grid(com, betas, cap_factors)
    df_grid.to_csv(out / "grid_beta_cap.csv", index=False)
    plot_sensitivity_summary(df_grid, df_beta, base_beta=base_beta, base_f=base_factor,
                             cap_base_kW=com.contract.capacity_limit_kW[0], path=str(out / "sensitivity_summary"))
    print(f"Tables and figures saved in {out}")


def run_settlement(com, res, out):
    """Write the member settlement and report budget balance and individual rationality."""
    from src.plotting import plot_settlement

    df = settle(com, res["R"], res["I"], res["C"])
    bb = budget_balance(res["C"])
    df.to_csv(out / "settlement.csv", index=False)

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
    print(f"\nTable and figure saved in {out}")


def run(com, out, *, sensitivity=False, settlement=False):
    """Run the R/I/C comparison and any selected Question 1 analyses."""
    from src.plotting import plot_import_vs_cap

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = com.contract.capacity_limit_kW

    results = {model: getattr(CommunityModel, model)(com).solve() for model in ("R", "I", "C")}
    metrics = {name: compute_metrics(res, results["R"], cap) for name, res in results.items()}

    # Table: one row per metric, one column per model
    print(f"{'Metric':<22}" + "".join(f"{name:>10}" for name in metrics))
    for key in metrics["R"]:
        print(f"{key:<22}" + "".join(f"{metrics[name][key]:>10.2f}" for name in metrics))
    print("Value of pooling (cost I - cost C, DKK):", round(results["I"]["cost"] - results["C"]["cost"], 2))
    print("Internal price C (DKK/kWh):", np.round(results["C"]["price"], 2))

    # Validation: model R against the reference values of the contract file
    print("--- Validation ---")
    print("  Model R:", {"daily_import_kWh": round(results["R"]["P_imp"].sum(), 2),
                         "peak_import_kW": round(results["R"]["P_imp"].max(), 2),
                         "cost_DKK": round(results["R"]["cost"], 2)})
    print("  File   :", com.contract.reference_model_R)

    # Check: the four components must add up to the total cost
    print("--- Cost breakdown check ---")
    for name, res in results.items():
        parts = res["cost_energy"] + res["cost_tariffs"] + res["cost_penalty"] + res["cost_own"]
        print(f"  {name}: total {res['cost']:.2f} | sum of parts {parts:.2f}")

    # Figure: import against the cap and internal price
    plot_import_vs_cap(results, com, str(out / "import_vs_cap"))
    print(f"Figure saved in {out}")

    if sensitivity:
        run_sensitivity(com, out)
    if settlement:
        run_settlement(com, results, out)
    return results
