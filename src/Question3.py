"""Evaluate fixed prices and train stochastic pricing policies for Question 3."""
import copy
import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src import DP as dp
from src.plotting import plot_Q3_part1_results
from src.scenario_generation import (
    day_data,
    draw_preference_scenarios,
    expected_preference_scenario,
    load_json,
    rules_to_spec,
    set_rule,
)


ACCOUNTING_TOL = 1e-4
METRICS_TOL = 1e-6
VARIANTS = ("SDP", "CVaR", "CVaR+CC")
PREFERENCE_RULE = {"lo": 0.8, "hi": 1.2, "mean": 1.0, "kappa": 2.0}


@dataclass(frozen=True)
class RiskSettings:
    """Cost risk and settlement limits for one stochastic pricing experiment."""
    confidence: float = 0.90
    cvar_weight: float = 0.50
    violation_probability: float = 0.10
    deficit_limit_DKK: float = 1.0
    ir_loss_limit_DKK: float = 1.0


def scenario_community(com, scenario):
    """Copy the community and update preferences; keep assets and contract fixed."""
    realized = copy.deepcopy(com)
    for d in realized.members:
        load = next(a for a in scenario["members"][d.consumer_id].values() if "load_id" in a)
        d.reference_load = d.load_max_kWh * np.asarray(load["reference_load_capacity_factor"], dtype=float)
        d.min_daily_energy_kWh = float(d.reference_load.sum())
        d.linear_disutility = float(load["linear_disutility_DKK_per_kWh"])
        d.load_lower_kWh = np.maximum(d.load_min_kWh, d.reference_load - d.max_hourly_deviation_kWh)
        d.load_upper_kWh = np.minimum(d.load_max_kWh, d.reference_load + d.max_hourly_deviation_kWh)
    return realized


def solve_independent_price_responses(com, fixed_prices, verbose=False):
    """Step A: find each member's minimum private cost at the fixed prices."""
    responses = {"net": {}, "own_cost": {}, "member_variables": {}, "optimal_member_cost": {}}
    for d in com.members:
        i, T = d.consumer_id, range(d.n_hours)
        prices = np.asarray(fixed_prices[i] if isinstance(fixed_prices, Mapping) else fixed_prices, dtype=float)
        if prices.shape != (d.n_hours,) or not np.isfinite(prices).all():
            raise ValueError(f"Invalid fixed prices for {i}")
        with gp.Model(f"response_{i}") as m:
            m.Params.OutputFlag = int(verbose)
            m.Params.FeasibilityTol = 1e-9
            m.Params.OptimalityTol = 1e-9
            v, _, net, own = dp.add_member(m, d, i)
            m.setObjective(own + gp.quicksum(float(prices[t]) * net[t] for t in T), GRB.MINIMIZE)
            m.optimize()
            if m.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Member {i}: response status {m.Status}")
            responses["net"][i] = np.array([net[t].getValue() for t in T])
            responses["own_cost"][i] = float(own.getValue())
            responses["optimal_member_cost"][i] = float(m.ObjVal)
            responses["member_variables"][i] = {name: np.array([series[t].X for t in T]) for name, series in v.items()}
    return responses


def solve_fixed_price_responses(com, fixed_prices, verbose=False, selection="optimistic", optimality_tol=1e-7):
    """Two-step response: private optima, then best/worst community cost among them.

    Prices are constants throughout. No budget-balance or IR constraints are added.
    The secondary model reuses C's physical constraints and Q2's exact meter
    accounting, but adds private-cost bounds so this is not centralized dispatch.
    A small absolute DKK tolerance permits numerical error in the private optima.
    """
    if selection not in ("optimistic", "pessimistic"):
        raise ValueError("selection must be 'optimistic' or 'pessimistic'.")
    if not np.isfinite(optimality_tol) or optimality_tol < 0:
        raise ValueError("optimality_tol must be finite and nonnegative.")
    independent = solve_independent_price_responses(com, fixed_prices, verbose)
    optimum = independent["optimal_member_cost"]

    physical = dp.CommunityModel.C(com)
    physical.verbose = verbose
    physical.build()
    m = physical.m
    try:
        m.Params.OutputFlag = int(verbose)
        m.Params.FeasibilityTol = 1e-9
        m.Params.OptimalityTol = 1e-9
        m.Params.MIPGap = 1e-9
        m.Params.MIPGapAbs = 1e-8
        m.Params.PreSOS1BigM = 0
        # Exact import/export and below/above-cap decomposition, also when maximizing.
        dp.DynamicPricingModel._enforce_meter_accounting(physical)
        private_cost = {}
        for i in com.member_ids:
            prices = np.asarray(fixed_prices[i] if isinstance(fixed_prices, Mapping) else fixed_prices, dtype=float)
            private_cost[i] = physical.own_cost[i] + gp.quicksum(float(prices[t]) * physical.net[i][t] for t in physical.T)
            m.addConstr(private_cost[i] <= optimum[i] + optimality_tol, name=f"best_response_{i}")

        objective = physical.cost_meter["community"] + gp.quicksum(physical.own_cost.values())
        m.setObjective(objective, GRB.MINIMIZE if selection == "optimistic" else GRB.MAXIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{selection} response selection: status {m.Status}")

        responses = {
            "net": {i: np.array([physical.net[i][t].getValue() for t in physical.T]) for i in com.member_ids},
            "own_cost": {i: float(physical.own_cost[i].getValue()) for i in com.member_ids},
            "member_variables": {i: {name: np.array([series[t].X for t in physical.T]) for name, series in physical.member_variables[i].items()} for i in com.member_ids},
            "optimal_member_cost": optimum,
            "selection": selection,
            "optimality_tol": optimality_tol,
            "community_objective": float(m.ObjVal),
            "runtime": float(m.Runtime),
            "member_optimality_gap": {i: float(private_cost[i].getValue() - optimum[i]) for i in com.member_ids},
        }
        for i, gap in responses["member_optimality_gap"].items():
            if gap > optimality_tol + 1e-6 or gap < -1e-6:
                raise AssertionError(f"Selected response is inconsistent with the private optimum: {i}, gap={gap:.3e}")
        return responses
    finally:
        m.dispose()


def part1_metrics(com, prices, cases, nominal=None, tol=1e-5):
    """One metrics function for nominal, profile-only and slope-only evaluations.

    Each case contains scenario, realized community, member responses and I costs.
    Frequencies are probability-weighted across draws; magnitudes are DKK except
    cap exceedance (kWh, hourly intervals). IR is checked separately per member.
    A positive IR gap is a loss against the same-scenario individual contract.
    Nominal comparisons use the SAME optimistic response selection rule.
    """
    if not cases or not np.isfinite(tol) or tol < 0:
        raise ValueError("Supply nonempty cases and a finite nonnegative tolerance.")
    weights = np.array([c["scenario"]["probability"] for c in cases], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Scenario probabilities must be nonnegative and sum to one.")
    weights /= weights.sum()
    ids = list(com.member_ids)
    rows = []
    for case, probability in zip(cases, weights):
        s, realized, response, outside = (case[k] for k in ("scenario", "community", "responses", "outside_option"))
        p = np.asarray(realized.energy_price, dtype=float)
        cap = np.broadcast_to(np.asarray(realized.contract.capacity_limit_kW, dtype=float), p.shape)
        total_net = np.sum([response["net"][i] for i in ids], axis=0)
        imports, exports = np.maximum(total_net, 0.0), np.maximum(-total_net, 0.0)
        below, above = np.minimum(imports, cap), np.maximum(imports - cap, 0.0)
        energy = float(p @ total_net)
        tariffs = float((1 - realized.contract.tariff_discount_ratio) * (realized.import_tariff * below.sum() + realized.export_tariff * exports.sum()))
        penalty = float(realized.contract.penalty_DKK_per_kWh * above.sum())
        own = float(sum(response["own_cost"].values()))
        bill = energy + tariffs + penalty
        members = {}
        for d in realized.members:
            i = d.consumer_id
            x = np.asarray(prices[i] if isinstance(prices, Mapping) else prices, dtype=float)
            v = response["member_variables"][i]
            payment = float(x @ response["net"][i])
            private_cost = float(response["own_cost"][i] + payment)
            gap = private_cost - float(outside["member_cost"][i])
            members[i] = {
                "theta_profile": float(s["theta"]["profile"][i]), "theta_cL": float(s["theta"]["cL"][i]),
                "reference_load": np.asarray(d.reference_load), "load_lower": np.asarray(d.load_lower_kWh), "load_upper": np.asarray(d.load_upper_kWh),
                "min_daily_energy_kWh": float(d.min_daily_energy_kWh), "disutility_DKK_per_kWh": float(d.linear_disutility),
                "load_kWh": float(v["load"].sum()), "net_kWh": float(response["net"][i].sum()),
                "payment_DKK": payment, "own_cost_DKK": float(response["own_cost"][i]), "cost_DKK": private_cost,
                "outside_cost_DKK": float(outside["member_cost"][i]), "IR_gap_DKK": gap, "IR_violation_DKK": max(gap, 0.0), "IR_violated": gap > tol,
                "net": response["net"][i], "variables": v,
            }
            if nominal is not None:
                base = nominal["members"][i]
                for key in ("min_daily_energy_kWh", "load_kWh", "net_kWh", "payment_DKK", "cost_DKK"):
                    members[i]["delta_" + key] = members[i][key] - base[key]
                members[i]["max_hourly_net_change_kWh"] = float(np.max(np.abs(response["net"][i] - base["net"])))
        balance = float(sum(d["payment_DKK"] for d in members.values()) - bill)
        row = {
            "scenario_id": s["id"], "probability": float(probability), "cost_DKK": bill + own,
            "energy_DKK": energy, "tariffs_DKK": tariffs, "penalty_DKK": penalty, "own_cost_DKK": own, "meter_bill_DKK": bill,
            "balance_DKK": balance, "BB_gap_DKK": abs(balance), "BB_violated": abs(balance) > tol,
            "deficit_DKK": max(-balance, 0.0), "deficit": balance < -tol,
            "IR_any_violated": any(d["IR_violated"] for d in members.values()),
            "IR_total_violation_DKK": sum(d["IR_violation_DKK"] for d in members.values()),
            "exceedance_kWh": float(above.sum()), "cap_exceeded": bool(np.any(above > tol)),
            "cap_hour_frequency": float(np.mean(above > tol)), "cap_max_hourly_exceedance_kWh": float(above.max()),
            "imports": imports, "exports": exports, "above": above, "members": members,
        }
        if "community_objective" in response and not np.isclose(row["cost_DKK"], response["community_objective"], atol=1e-5, rtol=1e-7):
            raise AssertionError("Optimistic objective and calculated community cost disagree.")
        if nominal is not None:
            row["delta_cost_DKK"] = row["cost_DKK"] - nominal["cost_DKK"]
            row["delta_balance_DKK"] = balance - nominal["balance_DKK"]
        rows.append(row)

    mean = lambda key: float(sum(w * r[key] for w, r in zip(weights, rows)))
    active = [r for r, w in zip(rows, weights) if w > 0]
    member_summary = {}
    for i in ids:
        member_summary[i] = {
            "mean_cost_DKK": float(sum(w * r["members"][i]["cost_DKK"] for w, r in zip(weights, rows))),
            "mean_IR_gap_DKK": float(sum(w * r["members"][i]["IR_gap_DKK"] for w, r in zip(weights, rows))),
            "IR_frequency": float(sum(w * r["members"][i]["IR_violated"] for w, r in zip(weights, rows))),
            "mean_IR_violation_DKK": float(sum(w * r["members"][i]["IR_violation_DKK"] for w, r in zip(weights, rows))),
            "max_IR_violation_DKK": max(r["members"][i]["IR_violation_DKK"] for r in active),
        }
    summary = {
        "scenarios": len(rows), "mean_cost_DKK": mean("cost_DKK"),
        "std_cost_DKK": float(np.sqrt(sum(w * (r["cost_DKK"] - mean("cost_DKK")) ** 2 for w, r in zip(weights, rows)))),
        "min_cost_DKK": min(r["cost_DKK"] for r in active), "max_cost_DKK": max(r["cost_DKK"] for r in active),
        "mean_energy_DKK": mean("energy_DKK"), "mean_tariffs_DKK": mean("tariffs_DKK"), "mean_penalty_DKK": mean("penalty_DKK"), "mean_own_cost_DKK": mean("own_cost_DKK"),
        "mean_balance_DKK": mean("balance_DKK"), "BB_frequency": mean("BB_violated"), "mean_BB_gap_DKK": mean("BB_gap_DKK"), "max_BB_gap_DKK": max(r["BB_gap_DKK"] for r in active),
        "deficit_frequency": mean("deficit"), "mean_deficit_DKK": mean("deficit_DKK"), "max_deficit_DKK": max(r["deficit_DKK"] for r in active),
        "IR_any_frequency": mean("IR_any_violated"), "mean_total_IR_violation_DKK": mean("IR_total_violation_DKK"),
        "IR_max_member_violation_DKK": max(r["members"][i]["IR_violation_DKK"] for r in active for i in ids),
        "IR_max_expected_violation_DKK": max(max(d["mean_IR_gap_DKK"], 0.0) for d in member_summary.values()),
        "cap_frequency": mean("cap_exceeded"), "cap_hour_frequency": mean("cap_hour_frequency"), "mean_exceedance_kWh": mean("exceedance_kWh"), "max_exceedance_kWh": max(r["exceedance_kWh"] for r in active),
    }
    if nominal is not None:
        summary.update(mean_delta_cost_DKK=mean("delta_cost_DKK"), mean_delta_balance_DKK=mean("delta_balance_DKK"))
    return {"summary": summary, "scenarios": rows, "members": member_summary, "tolerance": tol}


def evaluate_scenarios(com, prices, scenarios, nominal=None):
    """Keep prices fixed; apply preferences, solve I and optimistic responses."""
    start = perf_counter()
    cases, _ = prepare_cases(com, scenarios)
    for case in cases:
        case["responses"] = solve_fixed_price_responses(case["community"], prices)
    report = part1_metrics(com, prices, cases, nominal=nominal)
    report["summary"]["evaluation_seconds"] = perf_counter() - start
    return report


def print_part1_results(label, report):
    print(f"\n--- {label} ---")
    for key, value in report["summary"].items():
        print(f"{key:<36} {value:.1%}" if "frequency" in key else f"{key:<36} {value:.6f}")
    print("\nPer scenario: cost DKK | balance DKK | IR violation magnitude DKK | cap exceedance kWh")
    for r in report["scenarios"]:
        print(f"{str(r['scenario_id']):<9} {r['cost_DKK']:>10.4f} {r['balance_DKK']:>+12.4f} {r['IR_total_violation_DKK']:>12.4f} {r['exceedance_kWh']:>12.4f}")
    print(f"\n{'Member':<12} {'Mean IR gap':>12} {'IR freq':>9} {'Mean loss':>12} {'Max loss':>12}")
    for i, d in report["members"].items():
        print(f"{i:<12} {d['mean_IR_gap_DKK']:>+12.4f} {d['IR_frequency']:>9.1%} {d['mean_IR_violation_DKK']:>12.4f} {d['max_IR_violation_DKK']:>12.4f}")
    if "delta_cost_DKK" in report["scenarios"][0]:
        print(f"\n{'Draw':<6} {'Member':<12} {'theta P':>8} {'theta cL':>9} {'dEmin':>9} {'dLoad':>9} {'dPay':>9} {'dCost':>9}")
        for r in report["scenarios"]:
            for i, d in r["members"].items():
                print(f"{str(r['scenario_id']):<6} {i:<12} {d['theta_profile']:>8.4f} {d['theta_cL']:>9.4f} {d['delta_min_daily_energy_kWh']:>+9.4f} {d['delta_load_kWh']:>+9.4f} {d['delta_payment_DKK']:>+9.4f} {d['delta_cost_DKK']:>+9.4f}")


def save_part1_results(report, folder, name):
    """Only three outputs per experiment: full results, scenario CSV, member CSV."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(report, indent=2, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else a.item()), encoding="utf-8")
    for kind in ("scenarios", "members"):
        rows = []
        for scenario in report["scenarios"]:
            if kind == "scenarios":
                rows.append({k: v for k, v in scenario.items() if np.isscalar(v)})
            else:
                for i, member in scenario["members"].items():
                    rows.append({"scenario_id": scenario["scenario_id"], "probability": scenario["probability"], "member": i, **{k: v for k, v in member.items() if np.isscalar(v)}})
        with (folder / f"{name}_{kind}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def individual_contract(com):
    """Solve the individual contracts for the realized member preferences."""
    model = dp.CommunityModel.I(com).build()
    with model.m:
        return model.solve()


def solve_benchmark(com, outside, time_limit=None, verbose=False):
    """Solve the deterministic comparator for the stochastic pricing study."""
    start = perf_counter()
    model = dp.DynamicPricingModel(
        com, budget_balance=True, ir_baseline=outside["member_cost"], uniform_prices=False,
        time_limit=time_limit, mip_gap=1e-6, verbose=verbose, name="Q2_DPBBIR_IP",
    ).build()
    with model.m:
        build_seconds = perf_counter() - start
        result = model.solve()
        return result, solver_info(model.m, build_seconds)


def reproduce_question2(com, time_limit=None, verbose=False):
    """Solve the Part 1 benchmark, then check private optimality and accounting."""
    outside = individual_contract(com)
    # Keep the Part 1 solver settings: different optimal Q2 prices can give
    # different member responses when we later change their preferences.
    model = dp.DynamicPricingModel.DPBBIR_IP(com, ir_baseline=outside["member_cost"])
    model.time_limit = time_limit
    model.verbose = verbose
    model.build()
    with model.m:
        model.m.Params.MIPGap = 1e-6
        model.m.Params.MIPGapAbs = 1e-6
        q2 = model.solve()
    if q2.get("status") != GRB.OPTIMAL or "price" not in q2:
        raise RuntimeError(f"Q2 nominal validation requires an optimal solution (status {q2.get('status')}).")
    prices = copy.deepcopy(q2["price"])
    responses = solve_fixed_price_responses(com, prices)
    scenario = {"id": "nominal", "probability": 1.0, "theta": {q: {i: 1.0 for i in com.member_ids} for q in ("profile", "cL")}}
    case = {"scenario": scenario, "community": com, "responses": responses, "outside_option": outside}
    nominal_report = part1_metrics(com, prices, [case])
    nominal = nominal_report["scenarios"][0]
    # Recalculate saved Q2 dispatch using the SAME metrics function.
    saved = dict(responses, net=q2["net"], own_cost=q2["own_cost"])
    saved.pop("community_objective", None)
    saved_case = dict(case, responses=saved)
    accounting = part1_metrics(com, prices, [saved_case])["scenarios"][0]
    for metric, q2_key in (("cost_DKK", "cost"), ("meter_bill_DKK", "meter_bill"), ("balance_DKK", "manager_balance")):
        np.testing.assert_allclose(accounting[metric], q2[q2_key], atol=1e-5, rtol=1e-7, err_msg=f"Q2 accounting mismatch: {metric}")
    for i in com.member_ids:
        np.testing.assert_allclose(nominal["members"][i]["cost_DKK"], q2["member_cost"][i], atol=1e-5, rtol=1e-7, err_msg=f"Q2 member optimality mismatch: {i}")
    matches = np.isclose(nominal["cost_DKK"], q2["cost"], atol=1e-5, rtol=1e-7) and np.isclose(nominal["balance_DKK"], q2["manager_balance"], atol=1e-5, rtol=1e-7)
    print("\n--- Question 2 reproduction ---")
    print("Private optimality and saved-dispatch accounting: PASS")
    print(f"Community cost: Q2={q2['cost']:.6f}, nominal optimistic={nominal['cost_DKK']:.6f} DKK")
    print(f"Manager balance: Q2={q2['manager_balance']:+.6f}, nominal optimistic={nominal['balance_DKK']:+.6f} DKK")
    print("Nominal optimistic cost/BB reproduction:", "PASS" if matches else "DIFFERENT: hard BB/IR restrict Q2 selection; use nominal optimistic outcomes for scenario comparisons.")
    nominal_report["Q2_validation"] = {"accounting_pass": True, "member_optimality_pass": True, "optimistic_reproduction_pass": bool(matches), "Q2_cost_DKK": q2["cost"], "Q2_balance_DKK": q2["manager_balance"]}
    return prices, nominal_report


def scenario_inputs(com):
    """Use the same day, appliances and members as the loaded community."""
    raw = com.raw
    app, _ = day_data(raw["appliance_params"], raw["bus_params"], raw["weekly_profiles"], com.day)
    rules = load_json("preference_uncertainty_params.json")
    spec = rules_to_spec(rules, com.member_ids)
    set_rule(spec, "profile", **PREFERENCE_RULE)
    return app, raw["params"], rules, spec


def run_part1(com, folder, n_draws=5, time_limit=None, verbose=False):
    """Evaluate the Q2 prices on separate profile and disutility samples."""
    prices, nominal_report = reproduce_question2(com, time_limit, verbose)
    nominal = nominal_report["scenarios"][0]
    save_part1_results(dict(nominal_report, prices_DKK_per_kWh=prices), folder, "nominal")

    app, community, rules, spec = scenario_inputs(com)
    set_rule(spec, "cL", **PREFERENCE_RULE)
    settings = {
        "community": com.community, "day": com.day, "n_per_experiment": n_draws,
        "mean": PREFERENCE_RULE["mean"], "kappa": PREFERENCE_RULE["kappa"], "selection": "optimistic",
        "member_optimality_tolerance_DKK": 1e-7, "metrics_tolerance": 1e-5,
        "scenarios": {},
    }
    reports = {}
    for name, quantity, seed in (("profile", "profile", 2), ("disutility", "cL", 3)):
        scenarios = draw_preference_scenarios(
            app, community, rules, n=n_draws, seed=seed, quantities=(quantity,), spec=spec,
        )
        report = evaluate_scenarios(com, prices, scenarios, nominal)
        print_part1_results(f"{name.capitalize()} uncertainty: {n_draws} draws, seed={seed}", report)
        save_part1_results(report, folder, name)
        reports[name] = report
        settings[name] = {"seed": seed, "bounds": [PREFERENCE_RULE["lo"], PREFERENCE_RULE["hi"]]}
        settings["scenarios"][name] = [
            {key: scenario[key] for key in ("id", "probability", "theta", "members")}
            for scenario in scenarios
        ]

    write_json(folder / "scenarios.json", settings)
    plot_Q3_part1_results(reports["profile"], reports["disutility"], nominal_report, folder=folder / "figures")
    print(f"Part 1 results: {folder}")


class ScenarioBlock(dp.CommunityModel):
    """A scenario's member LPs and community meter in the shared model."""

    def __init__(self, model, com, tag, prices):
        super().__init__(com)
        self.m = model
        self.member_variables, self.member_constraints = {}, {}
        self.net, self.own_cost = {}, {}
        for member in com.members:
            i = member.consumer_id
            (self.member_variables[i], self.member_constraints[i],
             self.net[i], self.own_cost[i]) = dp.add_member(model, member, f"{tag}_{i}")

        p, contract = com.energy_price, com.contract
        cap = contract.capacity_limit_kW
        discount = 1 - contract.tariff_discount_ratio
        self.q_below = model.addVars(["community"], self.T, name=f"{tag}_below")
        self.q_above = model.addVars(["community"], self.T, name=f"{tag}_above")
        self.q_exp = model.addVars(["community"], self.T, name=f"{tag}_export")
        bill = gp.LinExpr()
        for t in self.T:
            below = self.q_below["community", t]
            above = self.q_above["community", t]
            export = self.q_exp["community", t]
            model.addConstr(below <= cap[t])
            model.addConstr(below + above - export == gp.quicksum(self.net[i][t] for i in self.ids))
            bill += (
                (p[t] + discount * com.import_tariff) * below
                + (p[t] + contract.penalty_DKK_per_kWh) * above
                - (p[t] - discount * com.export_tariff) * export
            )
        dp.DynamicPricingModel._enforce_meter_accounting(self)
        model.update()  # The Q2 KKT builder reads each member's matrix rows.
        self.private_cost = {
            i: dp.DynamicPricingModel._add_member_optimality(self, i, prices[i])
            for i in self.ids
        }
        own = gp.quicksum(self.own_cost.values())
        self.cost = bill + own
        self.balance = gp.quicksum(self.private_cost.values()) - self.cost

    def response(self):
        return {
            "net": {i: np.array([q.getValue() for q in self.net[i]]) for i in self.ids},
            "own_cost": {i: cost.getValue() for i, cost in self.own_cost.items()},
            "member_variables": {
                i: {name: np.array([series[t].X for t in self.T]) for name, series in groups.items()}
                for i, groups in self.member_variables.items()
            },
        }


def prepare_cases(com, scenarios):
    """Compute Model I once per scenario and reuse it across all policies."""
    cases = []
    preparation_seconds = outside_seconds = 0.0
    for scenario in scenarios:
        start = perf_counter()
        realized = scenario_community(com, scenario)
        preparation_seconds += perf_counter() - start
        start = perf_counter()
        outside_option = individual_contract(realized)
        outside_seconds += perf_counter() - start
        cases.append({"scenario": scenario, "community": realized, "outside_option": outside_option})
    return cases, {"preparation_seconds": preparation_seconds, "outside_option_seconds": outside_seconds}


def metrics(com, prices, cases, risk=RiskSettings()):
    """Add cost quantiles, CVaR and joint settlement violations to the common report."""
    report = part1_metrics(com, prices, cases, tol=METRICS_TOL)
    summary = report["summary"]
    # Weighted empirical quantile and upper-tail mean, including cutoff mass.
    rows = sorted(report["scenarios"], key=lambda row: row["cost_DKK"])
    weights = np.array([row["probability"] for row in rows])
    costs = np.array([row["cost_DKK"] for row in rows])
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    for cutoff, quantile_key, cvar_key in (
        (0.9, "p90_cost_DKK", "cvar90_cost_DKK"),
        (risk.confidence, "risk_quantile_cost_DKK", "risk_cvar_cost_DKK"),
    ):
        summary[quantile_key] = float(costs[np.searchsorted(cumulative, cutoff - 1e-12)])
        tail_weights = np.minimum(weights, np.maximum(cumulative - cutoff, 0.0))
        summary[cvar_key] = float(tail_weights @ costs / tail_weights.sum())
    summary["mean_cvar_score_DKK"] = (
        (1 - risk.cvar_weight) * summary["mean_cost_DKK"]
        + risk.cvar_weight * summary["risk_cvar_cost_DKK"]
    )

    for row in rows:
        row["cap_exceeded"] = row["exceedance_kWh"] > METRICS_TOL
        row["joint_settlement_excess_DKK"] = max(
            0.0, -row["balance_DKK"] - risk.deficit_limit_DKK,
            max(member["IR_gap_DKK"] for member in row["members"].values()) - risk.ir_loss_limit_DKK,
        )
        row["joint_settlement_violation"] = row["joint_settlement_excess_DKK"] > METRICS_TOL
    summary["cap_frequency"] = sum(
        row["probability"] for row in rows if row["cap_exceeded"]
    )
    summary["joint_violation_frequency"] = sum(
        row["probability"] for row in rows if row["joint_settlement_violation"]
    )
    summary["mean_joint_excess_DKK"] = sum(
        row["probability"] * row["joint_settlement_excess_DKK"] for row in rows
    )
    return report


def solver_info(model, build_seconds):
    """Collect the solver status, objective bounds, model size and timings."""
    def finite(value):
        return float(value) if abs(value) < GRB.INFINITY else None

    return {
        "status": model.Status,
        "optimal": model.Status == GRB.OPTIMAL,
        "runtime_seconds": model.Runtime,
        "build_seconds": build_seconds,
        "variables": model.NumVars,
        "constraints": model.NumConstrs,
        "sos1": model.NumSOS,
        "binary_variables": model.NumBinVars,
        "indicator_constraints": model.NumGenConstrs,
        "objective_bound_DKK": finite(model.ObjBound),
        "incumbent_DKK": model.ObjVal if model.SolCount else None,
        "mip_gap": finite(model.MIPGap) if model.SolCount else None,
    }


def solve_stochastic(com, cases, time_limit=None, verbose=False, *, variant="SDP",
                     risk=RiskSettings(), log_dir=Path("results/q3/solutionlogs"), label=None):
    """Use the same stochastic dispatch and settlement model for all variants."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    weights = [case["scenario"]["probability"] for case in cases]
    if not weights or min(weights) < 0 or not np.isclose(sum(weights), 1.0):
        raise ValueError("Scenario probabilities must be nonnegative and sum to one.")
    start = perf_counter()
    with gp.Model(f"Q3_{variant}") as model:

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        prefix = log_dir / f"Q3_{label or variant}_{len(cases)}_v3"
        model.Params.LogToConsole = int(verbose)
        model.Params.OutputFlag = 1
        model.Params.LogFile = f"{prefix}.log"
        model.Params.DisplayInterval = 1
        model.Params.SolFiles = f"{prefix}_incumbent"
        model.Params.PreSOS1BigM = 0
        model.Params.FeasibilityTol = 1e-6
        model.Params.MIPGap = 0.001
        if time_limit is not None:
            model.Params.TimeLimit = time_limit

        ids = com.member_ids
        hours = range(len(com.energy_price))
        prices = {i: model.addVars(hours, lb=-GRB.INFINITY, name=f"price_{i}") for i in ids}
        blocks = [
            ScenarioBlock(model, case["community"], f"s{s}", prices)
            for s, case in enumerate(cases)
        ]
        gaps = [
            {i: block.private_cost[i] - case["outside_option"]["member_cost"][i] for i in ids}
            for block, case in zip(blocks, cases)
        ]
        mean_cost = gp.quicksum(w * block.cost for w, block in zip(weights, blocks))
        objective = mean_cost
        if variant != "SDP":
            threshold = model.addVar(lb=-GRB.INFINITY, name="cvar_threshold")
            excess = model.addVars(len(blocks), lb=0, name="cost_excess")
            model.addConstrs(
                (excess[s] >= block.cost - threshold for s, block in enumerate(blocks)),
                name="cvar_excess",
            )
            cvar = threshold + gp.quicksum(w * excess[s] for s, w in enumerate(weights)) / (1 - risk.confidence)
            objective = (1 - risk.cvar_weight) * mean_cost + risk.cvar_weight * cvar
        model.setObjective(objective, GRB.MINIMIZE)
        model.addConstr(gp.quicksum(w * block.balance for w, block in zip(weights, blocks)) == 0,
                        name="expected_budget_balance")
        for i in ids:
            model.addConstr(gp.quicksum(w * gap[i] for w, gap in zip(weights, gaps)) <= 0,
                            name=f"expected_IR_{i}")

        if variant == "CVaR+CC":
            # One binary per scenario covers the manager and all members jointly.
            violation = model.addVars(len(blocks), vtype=GRB.BINARY, name="allow_violation")
            model.addConstr(gp.quicksum(w * violation[s] for s, w in enumerate(weights))
                            <= risk.violation_probability, name="joint_violation_budget")
            for s, (block, gap) in enumerate(zip(blocks, gaps)):
                model.addGenConstrIndicator(violation[s], False, -block.balance <= risk.deficit_limit_DKK)
                for i in ids:
                    model.addGenConstrIndicator(violation[s], False, gap[i] <= risk.ir_loss_limit_DKK)

        model.update()
        build_seconds = perf_counter() - start
        model.optimize()
        info = solver_info(model, build_seconds)
        if model.SolCount == 0:
            return None, None, info

        fixed_prices = {i: np.array([prices[i][t].X for t in hours]) for i in ids}
        solved_cases = [dict(case, responses=block.response()) for case, block in zip(cases, blocks)]
        report = metrics(com, fixed_prices, solved_cases, risk)
        for block, row in zip(blocks, report["scenarios"]):
            np.testing.assert_allclose(
                [row["cost_DKK"], row["balance_DKK"], *[row["members"][i]["cost_DKK"] for i in ids]],
                [block.cost.getValue(), block.balance.getValue(), *[block.private_cost[i].getValue() for i in ids]],
                atol=ACCOUNTING_TOL, rtol=0, err_msg=f"Scenario {row['scenario_id']} settlement",
            )
        summary = report["summary"]
        objective_key = "mean_cost_DKK" if variant == "SDP" else "mean_cvar_score_DKK"
        gap = model.ObjVal - summary[objective_key]
        if gap < -ACCOUNTING_TOL or (info["optimal"] and gap > ACCOUNTING_TOL):
            raise RuntimeError("Recomputed training objective disagrees with the solver objective.")
        if (abs(summary["mean_balance_DKK"]) > ACCOUNTING_TOL
                or summary["IR_max_expected_violation_DKK"] > ACCOUNTING_TOL):
            raise RuntimeError("Recomputed training dispatch fails expected BB or IR.")
        if variant == "CVaR+CC":
            violation_mass = sum(row["probability"] for row in report["scenarios"]
                                 if row["joint_settlement_excess_DKK"] > ACCOUNTING_TOL)
            if violation_mass > risk.violation_probability + METRICS_TOL:
                raise RuntimeError("Recomputed training dispatch exceeds the joint violation budget.")
        return fixed_prices, report, info


def solve_risk_neutral(com, cases, time_limit=None, verbose=False):
    """Keep the risk-neutral entry point available for existing callers."""
    return solve_stochastic(com, cases, time_limit, verbose)


def evaluate_prices(com, prices, cases, risk=RiskSettings()):
    """Use Step 1's optimistic best-response rule at a fixed posted price."""
    start = perf_counter()
    solved = [
        dict(case, responses=solve_fixed_price_responses(case["community"], prices))
        for case in cases
    ]
    report = metrics(com, prices, solved, risk)
    report["summary"]["evaluation_seconds"] = perf_counter() - start
    return report


def make_scenarios(com, max_train, n_test, train_seed, test_seed):
    """Draw one profile-only training pool and one separate test set."""
    app, raw_community, rules, spec = scenario_inputs(com)
    nominal = {
        "id": "nominal",
        "probability": 1.0,
        "theta": {q: dict.fromkeys(com.member_ids, 1.0) for q in ("profile", "cL")},
        "members": expected_preference_scenario(app, raw_community),
    }
    train = draw_preference_scenarios(app, raw_community, rules, max_train, train_seed, spec=spec)
    test = draw_preference_scenarios(app, raw_community, rules, n_test, test_seed, spec=spec)
    return nominal, train, test


def write_json(path, data):
    """Replace a progress file only after the new JSON has been written."""
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def comparison_rows(report):
    """Keep just the summary and member rows for the final console tables."""
    return {"summary": report["summary"], "members": report["members"]}


def print_comparison(title, reports):
    print(f"\n{title}")
    print(f"{'Metric':<37}" + "".join(f"{name:>16}" for name in reports))
    first = next(iter(reports.values()))["summary"]
    for key in first:
        print(f"{key:<37}" + "".join(f"{report['summary'][key]:>16.6f}" for report in reports.values()))
    print("\nMember IR (mean signed gap / violation frequency / mean positive loss)")
    print(f"{'Member':<12}" + "".join(f"{name:>28}" for name in reports))
    for i in next(iter(reports.values()))["members"]:
        cells = []
        for report in reports.values():
            row = report["members"][i]
            cells.append(f"{row['mean_IR_gap_DKK']:+.4f} / {row['IR_frequency']:.1%} / {row['mean_IR_violation_DKK']:.4f}")
        print(f"{i:<12}" + "".join(f"{cell:>28}" for cell in cells))


def experiment_matrix(n, sensitivity_size, risk, weights, probabilities):
    """Compare all models at each N and vary one risk parameter at the reference N."""
    for variant in VARIANTS:
        yield variant, variant, risk
    if n == sensitivity_size:
        for weight in sorted(set(weights) - {risk.cvar_weight}):
            yield f"CVaR_beta_{weight}", "CVaR", replace(risk, cvar_weight=weight)
        for probability in sorted(set(probabilities) - {risk.violation_probability}):
            yield (f"CC_epsilon_{probability}", "CVaR+CC",
                   replace(risk, violation_probability=probability))


def run(com, out, *, part="all", part1_draws=5, train_sizes=(5, 10, 12), test_size=100,
        train_seed=1, test_seed=99, time_limit=None, verbose=False,
        cvar_confidence=0.90, cvar_weight=0.50, violation_probability=0.10,
        deficit_limit=1.0, ir_loss_limit=1.0, sensitivity_size=10,
        weights=(0, 0.25, 0.75), probabilities=(0, 0.2, 0.3)):
    """Run either Q3 experiment or both, using the community supplied by main.py."""
    folder = Path(out)
    if part not in ("1", "2", "all"):
        raise ValueError("Choose part 1, 2, or all.")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("--time-limit must be positive and finite.")
    if part != "2" and part1_draws < 1:
        raise ValueError("--part1-draws must be positive.")
    if part == "1":
        run_part1(com, folder / "part1", part1_draws, time_limit, verbose)
        return

    sizes = sorted(set(train_sizes))
    if not sizes or sizes[0] < 1 or test_size < 1:
        raise ValueError("Training and test sizes must be positive.")
    if min(train_seed, test_seed) < 0:
        raise ValueError("Scenario seeds must be nonnegative.")
    if not (0 < cvar_confidence < 1 and 0 < cvar_weight <= 1 and 0 <= violation_probability < 1):
        raise ValueError("Check the CVaR confidence, reference weight and violation probability.")
    if any(not np.isfinite(x) or x < 0 for x in (deficit_limit, ir_loss_limit)):
        raise ValueError("Settlement limits must be finite and nonnegative.")
    if any(not 0 <= x <= 1 for x in weights) or any(not 0 <= x < 1 for x in probabilities):
        raise ValueError("Sensitivity weights must be in [0, 1] and probabilities in [0, 1).")
    if (weights or probabilities) and sensitivity_size not in sizes:
        raise ValueError("--sensitivity-size must be one of --train-sizes when running sensitivities.")
    risk = RiskSettings(cvar_confidence, cvar_weight, violation_probability, deficit_limit, ir_loss_limit)
    if part == "all":
        run_part1(com, folder / "part1", part1_draws, time_limit, verbose)

    folder.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    nominal, training_draws, test_draws = make_scenarios(com, max(sizes), test_size, train_seed, test_seed)
    generation_seconds = perf_counter() - start
    write_json(folder / "scenarios.json", {
        "community": com.community, "day": com.day, "train_seed": train_seed, "test_seed": test_seed,
        "train_sizes": sizes, "test_size": test_size,
        "profile_rule": PREFERENCE_RULE,
        "disutility_slope": "nominal",
        "train_pool": [dict(s, index=[[q, i, k] for (q, i), k in s["index"].items()]) for s in training_draws],
        "test": [dict(s, index=[[q, i, k] for (q, i), k in s["index"].items()]) for s in test_draws],
        "training_weights": "Each prefix of length N is reweighted to 1/N per scenario.",
    })

    nominal_cases, nominal_times = prepare_cases(com, [nominal])
    realized = nominal_cases[0]["community"]
    for base, actual in zip(com.members, realized.members):
        for attr in ("reference_load", "min_daily_energy_kWh", "load_lower_kWh",
                     "load_upper_kWh", "linear_disutility"):
            np.testing.assert_allclose(getattr(actual, attr), getattr(base, attr),
                                       atol=1e-9, rtol=0, err_msg=f"{base.consumer_id} {attr}")

    outside = nominal_cases[0]["outside_option"]
    q2, baseline_info = solve_benchmark(com, outside, time_limit, verbose)
    diagnostics = {
        "scenario_generation_seconds": generation_seconds, "nominal_preparation": nominal_times,
        "deterministic_solver": baseline_info, "risk_settings": asdict(risk),
    }
    write_json(folder / "diagnostics.json", diagnostics)
    if "price" not in q2:
        raise RuntimeError(f"Q2 produced no incumbent (status {baseline_info['status']}).")
    diagnostics["nominal_variants"] = {}

    for variant in VARIANTS:
        _, report, info = solve_stochastic(
            com, nominal_cases, time_limit, verbose, variant=variant, risk=risk,
            log_dir=folder / "solutionlogs", label=f"nominal_{variant}",
        )
        validation = "inconclusive"
        if baseline_info["optimal"] and info["optimal"]:
            difference = abs(info["incumbent_DKK"] - q2["cost"])
            print(f"Nominal objective difference: {difference:.6f} DKK")
            validation = "passed" if difference <= 0.5 else "inconclusive"  # stochastic solve uses a 0.1% relative gap
        diagnostics["nominal_variants"][variant] = {"solver": info, "validation": validation}
        if variant == "SDP":
            diagnostics.update(nominal_stochastic_solver=info, nominal_validation=validation)
            if report is not None:
                save_part1_results(report, folder, "nominal_stochastic")
        print(f"Nominal Q2 reproduction ({variant}): {validation}")
    write_json(folder / "diagnostics.json", diagnostics)

    train_cases, diagnostics["train_preparation"] = prepare_cases(com, training_draws)
    test_cases, diagnostics["test_preparation"] = prepare_cases(com, test_draws)
    det_test = evaluate_prices(com, q2["price"], test_cases, risk)
    save_part1_results(det_test, folder, "deterministic_test")
    study = {"risk_settings": asdict(risk), "deterministic_test": det_test["summary"]}
    policies = {"deterministic": {i: p.tolist() for i, p in q2["price"].items()}}
    write_json(folder / "prices.json", policies)
    file_tags = {"SDP": "sdp", "CVaR": "cvar", "CVaR+CC": "cvar_cc"}
    print("\nTraining the stochastic pricing models")
    for n in sizes:
        label = f"n{n}"
        prefix = [dict(case, scenario=dict(case["scenario"], probability=1 / n)) for case in train_cases[:n]]
        dp_train = evaluate_prices(com, q2["price"], prefix, risk)
        fits = {}
        train_replays = {"DP": comparison_rows(dp_train)}
        tests = {"DP": comparison_rows(det_test)}
        result = study[label] = {
            "variant_solvers": {}, "DP train": dp_train["summary"], "DP test": det_test["summary"],
        }
        save_part1_results(dp_train, folder, f"{label}_dp_train")
        for name, variant, settings in experiment_matrix(n, sensitivity_size, risk, weights, probabilities):
            prices, training, info = solve_stochastic(
                com, prefix, time_limit, verbose, variant=variant, risk=settings,
                log_dir=folder / "solutionlogs", label=name,
            )
            result["variant_solvers"][name] = dict(info, risk_settings=asdict(settings))
            if variant == "SDP":
                result["solver"] = info
            state = {GRB.OPTIMAL: "optimal", GRB.INFEASIBLE: "infeasible",
                     GRB.TIME_LIMIT: "time limit"}.get(info["status"], f"status {info['status']}")
            print(f"\nN={n}, {name}, {state}, runtime={info['runtime_seconds']:.2f}s, "
                  f"incumbent={info['incumbent_DKK']}, bound={info['objective_bound_DKK']}, gap={info['mip_gap']}")
            write_json(folder / "study.json", study)
            if prices is None:
                continue
            tag = file_tags.get(name, name.lower())
            price_key = label if variant == "SDP" else f"{label}_{tag}"
            policies[price_key] = {i: p.tolist() for i, p in prices.items()}
            write_json(folder / "prices.json", policies)
            for phase, cases, table in (("fit", None, fits), ("train", prefix, train_replays),
                                        ("test", test_cases, tests)):
                report = training if phase == "fit" else evaluate_prices(com, prices, cases, settings)
                save_part1_results(report, folder, f"{label}_{tag}_{phase}")
                result[f"{name} {phase}"] = report["summary"]
                table[name] = comparison_rows(report)
                write_json(folder / "study.json", study)
        if fits:
            print_comparison("Optimized in-sample dispatch", fits)
        print_comparison("In-sample fixed-price replay", train_replays)
        print_comparison("Out-of-sample fixed-price replay", tests)
        write_json(folder / "study.json", study)
        write_json(folder / "diagnostics.json", diagnostics)

    if len(policies) == 1:
        raise RuntimeError("The stochastic pricing solves produced no incumbents. See study.json.")
