"""
Question 3: The base stochastic model
"""
import copy
import json
import logging
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from src import DP as dp
from src import data_loader as dl
from src.Question3_step1 import scenario_community
from src.scenario_generation import draw_preference_scenarios, expected_preference_scenario, load_json

logger = logging.getLogger(__name__)

# ---Scenario Data; From joint scenarios to model inputs---

def nominal_scenarios(community):
    members = expected_preference_scenario(community.raw["appliance_params"], community.raw["params"])  
    theta = {q: {i: 1.0 for i in community.member_ids} for q in ("profile", "cL")}
    return {"id": "nominal", "probability": 1.0, "theta": theta, "members": members}

def scenario_inputs(community, scenarios):
    probs = np.array([s["probability"] for s in scenarios], dtype=float)
    if not np.isclose(probs.sum(), 1.0) or (probs < 0).any():
        raise ValueError("Scenario probabilities must be non-negative and sum to 1.")
    out = []
    for s in scenarios:
        community_s = scenario_community(community, s)
        outside = dp.CommunityModel.I(community_s).solve()["member_cost"]
        out.append({"id": s["id"], "pi": float(s["probability"]), "com": community_s, "outside": outside})
        logger.info("Scenario inputs ready: %d scenarios", len(out))
    return out

# ---MODEL SDP---
class StochasticPricingModel(dp.DynamicPricingModel):
    """
    SDP: scenario-indexed extention of DynamicPricingModel of Question2
    -> Inherits add_member, _add_member_optimality and _internal_prices from DynamicPricingModel
    """
    OBJECTIVES = ("expected",)
    CONSTRAINTS = ("expectation",) 

    def __init__(self, community, scenarios, *, objective="expected", constraints="expectation",
                 budget_balance=True, individual_rationality=True, uniform_prices=False,
                 time_limit=None, mip_gap=None, log_file=None, verbose=False, name="SDP"):
        if objective not in self.OBJECTIVES:
            raise NotImplementedError(f"objective '{objective}' comes in step 3; available: {self.OBJECTIVES}")
        if constraints not in self.CONSTRAINTS:
            raise NotImplementedError(f"constraints '{constraints}' come in step 3; available: {self.CONSTRAINTS}")
        super().__init__(community, budget_balance=budget_balance, ir_baseline=None, uniform_prices=uniform_prices,
                         time_limit=time_limit, mip_gap=mip_gap, verbose=verbose, name=name)
        self.objective = objective
        self.constraints = constraints
        self.individual_rationality = individual_rationality
        self.log_file = log_file
        self.scen = scenario_inputs(community, scenarios)     # per scenario: id, pi, community, outside option
        self.S = range(len(self.scen))
        logger.info("%s set up: %d scenarios, %d members, objective '%s', constraints '%s', %s prices",
                    self.name, len(self.S), len(self.ids), objective, constraints,
                    "uniform" if uniform_prices else "per-member")

    def _add_scenario(self, s):
        """Members (with the scenario's preferences) and the meter of scenario s."""
        sc, m, T = self.scen[s], self.m, self.T
        for d in sc["com"].members:
            key = (d.consumer_id, s)                                    # member i in scenario s
            v, c, net, own = dp.add_member(m, d, f"{d.consumer_id}_s{s}")
            self.member_variables[key], self.member_constraints[key] = v, c
            self.net[key], self.own_cost[key] = net, own
        cap, contract = self.com.contract.capacity_limit_kW, self.com.contract
        below = m.addVars(T, lb=0, ub=[float(c) for c in cap], name=f"q_below_s{s}")   # import within the cap
        above = m.addVars(T, lb=0, name=f"q_above_s{s}")                                # import above the cap
        export = m.addVars(T, lb=0, name=f"q_exp_s{s}")                                 # export
        m.addConstrs((below[t] + above[t] - export[t] == gp.quicksum(self.net[(i, s)][t] for i in self.ids)
                      for t in T), name=f"balance_s{s}")
        for t in T:                                                 # exact meter accounting, as in Question 2
            total_import = m.addVar(lb=0, name=f"meter_import_s{s}_{t}")
            unused_cap = m.addVar(lb=0, name=f"meter_unused_cap_s{s}_{t}")
            m.addConstr(total_import == below[t] + above[t])
            m.addConstr(unused_cap == float(cap[t]) - below[t])
            m.addSOS(GRB.SOS_TYPE1, [total_import, export[t]])
            m.addSOS(GRB.SOS_TYPE1, [unused_cap, above[t]])
        beta, alpha = contract.tariff_discount_ratio, contract.penalty_DKK_per_kWh
        p, ti, te = self.com.energy_price, self.com.import_tariff, self.com.export_tariff
        self.q_below[s], self.q_above[s], self.q_exp[s] = below, above, export
        self.bill[s] = gp.quicksum((p[t] + (1 - beta) * ti) * below[t] + (p[t] + alpha) * above[t]
                                   - (p[t] - (1 - beta) * te) * export[t] for t in T)
        logger.debug("  scenario %s added (pi = %.3f)", sc["id"], sc["pi"])


    def build(self):
        self.m = gp.Model(self.name)
        self.m.Params.OutputFlag = int(self.verbose)
        self.m.Params.PreSOS1BigM = 0                     # keep SOS1 as SOS1 (no hidden big-M), as in Question 2
        if self.log_file:
            self.m.Params.LogFile = str(self.log_file)    # Gurobi's own log, even when OutputFlag = 0
        if self.time_limit is not None:
            self.m.Params.TimeLimit = self.time_limit
        if self.mip_gap is not None:
            self.m.Params.MIPGap = self.mip_gap
        self.member_variables, self.member_constraints, self.net, self.own_cost = {}, {}, {}, {}
        self.q_below, self.q_above, self.q_exp, self.bill = {}, {}, {}, {}

        for s in self.S:
            self._add_scenario(s)

        if self.uniform_prices:
            x = self.m.addVars(self.T, lb=-GRB.INFINITY, name="posted_price")
            self.posted_prices = dict.fromkeys(self.ids, x)
        else:
            self.posted_prices = {i: self.m.addVars(self.T, lb=-GRB.INFINITY, name=f"posted_price_{i}")
                                  for i in self.ids}
        self.m.update()

        self.member_total_cost = {(i, s): self._add_member_optimality((i, s), self.posted_prices[i])
                                  for s in self.S for i in self.ids}
        self.member_payment = {k: self.member_total_cost[k] - self.own_cost[k] for k in self.member_total_cost}

        pi = {s: self.scen[s]["pi"] for s in self.S}
        self.scenario_cost = {s: self.bill[s] + gp.quicksum(self.own_cost[(i, s)] for i in self.ids) for s in self.S}
        self.balance_s = {s: gp.quicksum(self.member_payment[(i, s)] for i in self.ids) - self.bill[s] for s in self.S}
        if self.budget_balance:
            self.m.addConstr(gp.quicksum(pi[s] * self.balance_s[s] for s in self.S) == 0, name="BB_expectation")
        if self.individual_rationality:
            for i in self.ids:
                self.m.addConstr(gp.quicksum(pi[s] * (self.member_total_cost[(i, s)] - self.scen[s]["outside"][i])
                                             for s in self.S) <= 0, name=f"IR_expectation_{i}")
        self.m.setObjective(gp.quicksum(pi[s] * self.scenario_cost[s] for s in self.S), GRB.MINIMIZE)
        self.m.update()
        logger.info("%s built: %d variables, %d constraints, %d SOS1 sets", self.name,
                    self.m.NumVars, self.m.NumConstrs, self.m.NumSOS)
        return self

    def solve(self):
        if not hasattr(self, "m"):
            self.build()
        logger.info("%s solving ...", self.name)
        self.m.optimize()
        res = {"status": self.m.Status, "runtime": self.m.Runtime,
               "size": {"variables": self.m.NumVars, "constraints": self.m.NumConstrs, "SOS1": self.m.NumSOS,
                        "scenarios": len(self.S), "members": len(self.ids)}}
        if self.m.Status == GRB.TIME_LIMIT:
            logger.warning("%s stopped at the time limit (%.0f s)", self.name, self.m.Runtime)
        if self.m.SolCount == 0:
            logger.error("%s: no solution found (status %s)", self.name, self.m.Status)
            return res
        res["mip_gap"] = self.m.MIPGap
        res["objective"] = self.m.ObjVal                                   # expected community cost (DKK)
        res["price"] = self._internal_prices()
        rows = []
        for s in self.S:
            pay = {i: self.member_payment[(i, s)].getValue() for i in self.ids}
            cost = {i: self.member_total_cost[(i, s)].getValue() for i in self.ids}
            ir_gap = {i: cost[i] - self.scen[s]["outside"][i] for i in self.ids}
            rows.append({"id": self.scen[s]["id"], "pi": self.scen[s]["pi"],
                         "cost": self.scenario_cost[s].getValue(), "bill": self.bill[s].getValue(),
                         "exceedance_kWh": float(sum(self.q_above[s][t].X for t in self.T)),
                         "balance": self.balance_s[s].getValue(), "member_cost": cost, "member_payment": pay,
                         "IR_gap": ir_gap, "IR_violations": sum(g > 1e-6 for g in ir_gap.values())})
        res["scenarios"] = rows
        res["expected_balance"] = sum(r["pi"] * r["balance"] for r in rows)
        res["expected_IR_gap"] = {i: sum(r["pi"] * r["IR_gap"][i] for r in rows) for i in self.ids}
        logger.info("%s: objective %.4f DKK, E[balance] %+.4f DKK, status %s, gap %.1e, %.1f s", self.name,
                    res["objective"], res["expected_balance"], res["status"], res["mip_gap"], res["runtime"])
        for r in rows:
            logger.debug("  scenario %s: cost %.2f, bill %.2f, exceedance %.2f kWh, balance %+.2f, IR violations %d",
                         r["id"], r["cost"], r["bill"], r["exceedance_kWh"], r["balance"], r["IR_violations"])
        return res    

def validate_limit_case(community, time_limit=600, out=Path("results/q3")):
    """N = 1, theta = 1: SDP^E and DPBBIR_IP are the same problem, so their optimal costs must coincide."""
    logger.info("Limit case: Question 2 model against SDP^E with one scenario at theta = 1")
    outside = dp.CommunityModel.I(community).solve()["member_cost"]
    q2 = dp.DynamicPricingModel.DPBBIR_IP(community, ir_baseline=outside)
    q2.time_limit = time_limit
    r2 = q2.solve()
    sdp = StochasticPricingModel(community, [nominal_scenarios(community)], time_limit=time_limit,
                                 log_file=out / "gurobi_limit_case.log")
    r3 = sdp.solve()
    logger.info("Q2 DPBBIR_IP : cost %.4f DKK, balance %+.4f, status %s, gap %.1e, %.1f s",
                r2["cost"], r2["manager_balance"], r2["status"], r2["mip_gap"], r2["runtime"])
    logger.info("SDP^E, N = 1 : cost %.4f DKK, balance %+.4f, status %s, gap %.1e, %.1f s",
                r3["objective"], r3["expected_balance"], r3["status"], r3["mip_gap"], r3["runtime"])
    diff = abs(r2["cost"] - r3["objective"])
    tol = 1e-4 * max(1.0, abs(r2["cost"]))
    (logger.info if diff <= tol else logger.warning)("difference   : %.2e DKK (%s) | size %s", diff,
                                                     "PASS" if diff <= tol else "CHECK", r3["size"])
    return r2, r3

#---solve SDP^E on the in-sample sets (nested: the first n of one draw)---
def draw_in_sample(community, n_max=10, seed=1, quantities=("profile",)):
    """One draw of n_max joint scenarios with the rules of the uncertainty file (base study: profile only)."""
    rules = load_json("preference_uncertainty_params.json")
    scenarios = draw_preference_scenarios(community.raw["appliance_params"], community.raw["params"], rules,
                                          n=n_max, seed=seed, quantities=quantities)
    logger.info("In-sample draw: %d joint scenarios, seed %d, uncertain: %s", n_max, seed, ", ".join(quantities))
    return scenarios


def nested(scenarios, n):
    """The first n scenarios of a draw, each with probability 1/n."""
    return [dict(s, probability=1.0 / n) for s in scenarios[:n]]


def solve_in_sample(community, sizes=(1, 2, 5, 10), seed=1, time_limit=1800, out=Path("results/q3")):
    """SDP^E for every in-sample size in `sizes`, on nested sets. Saves the prices and a summary table."""
    draw = draw_in_sample(community, n_max=max(sizes), seed=seed)
    rows, prices = [], {}
    for n in sizes:
        name = f"SDP_E_N{n}"
        res = StochasticPricingModel(community, nested(draw, n), name=name, time_limit=time_limit,
                                     log_file=out / f"gurobi_{name}.log").solve()
        row = {"N": n, "status": res["status"], "runtime_s": res["runtime"], **res["size"]}
        if "objective" in res:
            sc = res["scenarios"]
            row.update(objective_DKK=res["objective"], mip_gap=res["mip_gap"],
                       expected_balance_DKK=res["expected_balance"],
                       max_scenario_deficit_DKK=max(max(-r["balance"], 0.0) for r in sc),
                       mean_exceedance_kWh=sum(r["pi"] * r["exceedance_kWh"] for r in sc),
                       scenarios_with_IR_violation=sum(r["IR_violations"] > 0 for r in sc))
            prices[n] = res["price"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("N")
    table.to_csv(out / "in_sample_SDP_E.csv")
    with open(out / "prices_SDP_E.json", "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "sizes": list(sizes),
                   "prices": {str(n): {i: np.asarray(v).tolist() for i, v in p.items()} if isinstance(p, dict)
                              else np.asarray(p).tolist() for n, p in prices.items()}}, f, indent=1)
    logger.info("In-sample results:\n%s", table.round(3).to_string())
    return table, prices

if __name__ == "__main__":
    out = Path("results/q3")
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out / "step2.log", mode="w", encoding="utf-8")],
    )
    com = dl.load_community("base", day="d1")
    sdp = StochasticPricingModel(com, [nominal_scenarios(com)]) 
    res = sdp.solve()
    validate_limit_case(com, out=out)
    solve_in_sample(com, sizes=(1, 2, 5, 10), seed=1, time_limit=1800, out=out)
