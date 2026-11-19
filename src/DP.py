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
        self.member_variables, self.member_constraints = {}, {}
        for d in self.com.members:
            v, c, net, own_cost = add_member(self.m, d, d.consumer_id)
            self.member_variables[d.consumer_id] = v
            self.member_constraints[d.consumer_id] = c
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

        return self._collect_results()

    def _collect_results(self):
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
        else:
            # Dual of the internal balance: the internal price of central dispatch
            res["price"] = self._internal_prices()
        return res

    def _internal_prices(self):
        return np.array([self.balance["community", t].Pi for t in self.T])


class DynamicPricingModel(CommunityModel):
    """Q2 optimistic MILP with a soft DSO cap and SOS1 member KKT conditions.

    ir_baseline maps member IDs to outside-option costs in DKK.
    A partial mapping imposes IR only on the listed members.
    """
        
    def __init__(self, com, *, budget_balance=False, ir_baseline=None,
                     uniform_prices=True, time_limit=None, mip_gap=None,
                     verbose=False, name='DP'):
        # Parameters after * must be passed by name, e.g. budget_balance=True.

        super().__init__(com, name=name, verbose=verbose)
        # Initialize the parent class with the community data and model name.

        self.budget_balance = budget_balance
        # Store whether the manager must collect exactly its external bill.
            
        self.ir_baseline = {} if ir_baseline is None else dict(ir_baseline)
        # Store each member's outside-option cost. Empty means no IR constraints.

        unknown = self.ir_baseline.keys() - set(self.ids)
        # Find baseline entries whose member IDs do not exist in the community.

        if unknown:
            raise ValueError(f"Unknown IR member IDs: {sorted(unknown)}")
        # Stop if any baseline refers to an unknown member.

        self.uniform_prices = uniform_prices
        # True gives everyone the same price vector.

        self.time_limit = time_limit
        # Store the optional maximum solving time in seconds.

        self.mip_gap = mip_gap
        # Store the optional relative optimality gap.
        self.name = name
        
    @classmethod
    def DP(cls, com):
        """DP model: with signals and uniforms pricing"""
        return cls(com, budget_balance=False, ir_baseline=None,
                         uniform_prices=True, verbose=False, name="DP")


    @classmethod
    def DPBB(cls, com):
        """DPx model: with signals, BB and uiniform prices"""
        return cls(com, budget_balance=True, ir_baseline=None,
                         uniform_prices=True, verbose=False, name="DPBB")

    @classmethod
    def DPBBIR_IP(cls, com, ir_baseline):
        """DPx model: with signals, BB + IR and individual prices"""
        return cls(com, budget_balance=True, ir_baseline=ir_baseline,
                         uniform_prices=False, verbose=False, name="DPBBIR_IP")

    def _add_member_optimality(self, member_id, prices):
        """Attach the KKT system to the existing member LP and return its dual value."""
        m = self.m
        variables = [
            var
            for series in self.member_variables[member_id].values()
            for var in series.values()
        ]
        # Flatten this member's variable groups into one list:
        # hourly load, deviations, PV, and battery variables where applicable.

        stationarity = {
            var.index: gp.LinExpr() for var in variables
        }
        # Start an empty stationarity expression for every household variable.
        # var.index identifies the variable inside the Gurobi model.

        own = self.own_cost[member_id]
        # Retrieve PV operating cost plus load disutility.

        dual_value = gp.LinExpr(own.getConstant())
        # Start the dual objective with the constant part of the own-cost expression.

        for k in range(own.size()):
            stationarity[own.getVar(k).index].addConstant(
                own.getCoeff(k)
            )
        # Add each own-cost coefficient to the corresponding derivative.

        for t, net in enumerate(self.net[member_id]):
            # Loop through the member's hourly net-exchange expressions.

            dual_value += net.getConstant() * prices[t]
            # Include any constant contribution from the payment x_t * net_t.

            for k in range(net.size()):
                stationarity[net.getVar(k).index] += (
                    net.getCoeff(k) * prices[t]
                )
            # Add the derivative of the hourly payment to stationarity.

        def add_row(lhs, sense, rhs, name):
            # Use RHS-sensitivity duals: >= has positive sign, <= negative.
            if sense == GRB.EQUAL:
                dual = m.addVar(lb=-GRB.INFINITY, name=f"dual_{name}")
            else:
                sign = 1 if sense == GRB.GREATER_EQUAL else -1
                magnitude = m.addVar(lb=0, name=f"dual_abs_{name}")
                slack = m.addVar(lb=0, name=f"slack_{name}")
                m.addConstr(slack == sign * (lhs - rhs), name=f"slack_def_{name}")
                m.addSOS(GRB.SOS_TYPE1, [slack, magnitude])
                dual = sign * magnitude
            for k in range(lhs.size()):
                stationarity[lhs.getVar(k).index] -= lhs.getCoeff(k) * dual
            return rhs * dual

        for group in self.member_constraints[member_id].values():
            rows = group.values() if isinstance(group, gp.tupledict) else (group,)
            for row in rows:
                dual_value += add_row(m.getRow(row), row.Sense, row.RHS, row.ConstrName)

        # Native bounds are not in getRow(). Include their duals explicitly.
        for var in variables:
            lhs = gp.LinExpr([1.0], [var])
            if var.LB > -GRB.INFINITY:
                dual_value += add_row(lhs, GRB.GREATER_EQUAL, var.LB, f"{var.VarName}_lb")
            if var.UB < GRB.INFINITY:
                dual_value += add_row(lhs, GRB.LESS_EQUAL, var.UB, f"{var.VarName}_ub")
            m.addConstr(stationarity[var.index] == 0, name=f"stationarity_{var.VarName}")
        return dual_value

    def _enforce_meter_accounting(self):
        """Make the external bill a function of the net exchange even with BB."""
        m = self.m
        cap = self.com.contract.capacity_limit_kW
        for t in self.T:
            below = self.q_below["community", t]
            above = self.q_above["community", t]
            export = self.q_exp["community", t]
            total_import = m.addVar(lb=0, name=f"meter_import_{t}")
            unused_cap = m.addVar(lb=0, name=f"meter_unused_cap_{t}")
            m.addConstr(total_import == below + above)
            m.addConstr(unused_cap == cap[t] - below)
            m.addSOS(GRB.SOS_TYPE1, [total_import, export])
            m.addSOS(GRB.SOS_TYPE1, [unused_cap, above])

    def build(self):
        super().build()
        
        self._enforce_meter_accounting()
        
        if self.time_limit is not None:
            self.m.Params.TimeLimit = self.time_limit
        if self.mip_gap is not None:
            self.m.Params.MIPGap = self.mip_gap
        # Leave SOS1 branching to Gurobi without converting it to big-M rows.
        self.m.Params.PreSOS1BigM = 0
        
        if self.uniform_prices:
            prices = self.m.addVars(self.T, lb=-GRB.INFINITY, name="posted_price")
            self.posted_prices = dict.fromkeys(self.ids, prices)
        else: # If price for every member
            self.posted_prices = {
                i: self.m.addVars(self.T, lb=-GRB.INFINITY, name=f"posted_price_{i}")
                for i in self.ids
                }
            
        self.m.update()
        
        self.member_total_cost = {
            i: self._add_member_optimality(i, self.posted_prices[i]) for i in self.ids
        }
        
        self.member_payment = {
            i: self.member_total_cost[i] - self.own_cost[i] for i in self.ids
        }
        if self.budget_balance:
            self.m.addConstr(
                gp.quicksum(self.member_payment.values()) == self.cost_meter["community"],
                name="manager_budget_balance",
            )
        for i, outside_cost in self.ir_baseline.items():
            self.m.addConstr(
                self.member_total_cost[i] <= outside_cost, name=f"individual_rationality_{i}"
            )
        return self

    def _internal_prices(self):
        if self.uniform_prices:
            return np.array([self.posted_prices[self.ids[0]][t].X for t in self.T])
        return {
            i: np.array([self.posted_prices[i][t].X for t in self.T]) for i in self.ids
        }

    def solve(self):
        """Return status and, if available, an incumbent with settlement results."""
        if not hasattr(self, "m"):
            self.build()
        self.m.optimize()
        info = {"status": self.m.Status, "runtime": self.m.Runtime}
        if self.m.SolCount == 0:
            return info

        res = self._collect_results()
        res.update(info, mip_gap=self.m.MIPGap)
        prices = dict.fromkeys(self.ids, res["price"]) if self.uniform_prices else res["price"]
        res["member_payment"] = {i: float(prices[i] @ res["net"][i]) for i in self.ids}
        res["member_cost"] = {
            i: res["own_cost"][i] + res["member_payment"][i] for i in self.ids
        }
        res["max_member_duality_gap"] = max(
            abs(res["member_cost"][i] - self.member_total_cost[i].getValue())
            for i in self.ids
        )
        res["meter_bill"] = self.cost_meter["community"].getValue()
        res["manager_balance"] = sum(res["member_payment"].values()) - res["meter_bill"]
        res["saving_vs_baseline"] = {
            i: outside_cost - res["member_cost"][i]
            for i, outside_cost in self.ir_baseline.items()
        }
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
        "congestion_removed": 1 - exceedance / exceedance_R,
        "hours_above": int((P > cap + 1e-6).sum()),
        "peak": float(P.max()),
        "peak_reduction": float(1 - P.max() / P_R.max()),
        "tariff_revenue_lost": res_R["cost_tariffs"] - res["cost_tariffs"],
    }
    


if __name__ == "__main__":
    from src.plotting import plot_import_vs_cap

    com = load_community("base", day="d1")
    cap = com.contract.capacity_limit_kW

    results = {f.__name__: f(com).solve() for f in (CommunityModel.R, CommunityModel.I, CommunityModel.C)}
    print("Solved R I C models")
    models = {
            "DP": DynamicPricingModel.DP(com),
            "DPBB": DynamicPricingModel.DPBB(com),
            "DPBBIR_IP": DynamicPricingModel.DPBBIR_IP(
                com,
                ir_baseline=results["I"]["member_cost"],
            ),
        }

    results.update({
            name: model.solve()
            for name, model in models.items()
        })

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
    plot_import_vs_cap(results, com, "results/q1/import_vs_cap")
    print("Figure saved in results/q1/")
    
    