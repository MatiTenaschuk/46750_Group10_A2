"""Question 3 -- Step 1 only. Run from the existing project root:
    python Question3_step1.py

Requires the original src/DP.py, src/data_loader.py, src/scenario_generation.py
and data/ JSON files. Uses the same Q2 DPBBIR_IP model as the previous runner.
One nominal validation, five profile-only draws, five slope-only draws.
No stochastic pricing model, training/test split or risk-model registry here.
"""
import copy
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src import DP as dp
from src import data_loader as dl
from src.scenario_generation import load_json, day_data, rules_to_spec, set_rule, draw_preference_scenarios
from src.plotting import plot_Q3_part1_results

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
    cases = []
    for scenario in scenarios:
        realized = scenario_community(com, scenario)
        outside = dp.CommunityModel.I(realized).solve()
        responses = solve_fixed_price_responses(realized, prices)
        cases.append({"scenario": scenario, "community": realized, "responses": responses, "outside_option": outside})
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


def reproduce_question2(com):
    """Solve original Q2 and check private optima, accounting and nominal selection.

    Exact dispatch need not be unique. Report a nominal cost/BB mismatch honestly:
    Q2 hard BB/IR can restrict selection beyond the teacher's optimistic rule.
    """
    outside = dp.CommunityModel.I(com).solve()
    model = dp.DynamicPricingModel.DPBBIR_IP(com, ir_baseline=outside["member_cost"])
    try:
        model.build()
        model.m.Params.MIPGap = 1e-6
        model.m.Params.MIPGapAbs = 1e-6
        q2 = model.solve()
        if q2.get("status") != GRB.OPTIMAL or "price" not in q2:
            raise RuntimeError(f"Q2 not optimal: {q2.get('status')}")
    finally:
        if hasattr(model, "m"):
            model.m.dispose()
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


if __name__ == "__main__":
    # 1. REPRODUCE QUESTION 2; KEEP ITS PRICES FIXED FOR ALL SUBSEQUENT DRAWS.
    com = dl.load_community("base", day="d1")
    prices, nominal_report = reproduce_question2(com)
    nominal = nominal_report["scenarios"][0]
    save_part1_results(nominal_report, "results/q3_step1", "nominal")

    # 2. LOAD THE PROVIDED GENERATOR AND SET TWO SEPARATE EXPERIMENTS.
    app = load_json("appliance_params.json")
    community = load_json("params_community_base.json")
    rules = load_json("preference_uncertainty_params.json")
    week, bus = load_json("weekly_profiles.json"), load_json("bus_params.json")
    app_day, _ = day_data(app, bus, week, "d1")
    spec = rules_to_spec(rules, com.member_ids)
    set_rule(spec, "profile", lo=0.8, hi=1.2, mean=1.0, kappa=2.0)
    set_rule(spec, "cL", lo=0.8, hi=1.2, mean=1.0, kappa=2.0)

    # 3. FIVE PROFILE-ONLY DRAWS: SLOPES STAY NOMINAL.
    profile_scenarios = draw_preference_scenarios(app_day, community, rules, n=5, seed=2, quantities=("profile",), spec=copy.deepcopy(spec))
    profile_results = evaluate_scenarios(com, prices, profile_scenarios, nominal)
    print_part1_results("Profile uncertainty: 5 draws, seed=1", profile_results)
    save_part1_results(profile_results, "results/q3_step1", "profile")

    # 4. FIVE SLOPE-ONLY DRAWS: PROFILES/DAILY REQUIREMENTS STAY NOMINAL.
    slope_scenarios = draw_preference_scenarios(app_day, community, rules, n=5, seed=3, quantities=("cL",), spec=copy.deepcopy(spec))
    slope_results = evaluate_scenarios(com, prices, slope_scenarios, nominal)
    print_part1_results("Disutility uncertainty: 5 draws, seed=42", slope_results)
    save_part1_results(slope_results, "results/q3_step1", "disutility")

    # Save draws/settings as reproducible inputs; tuple-key index traces become lists.
    scenario_data = {name: [{"id": s["id"], "probability": s["probability"], "theta": s["theta"], "members": s["members"]} for s in scenarios] for name, scenarios in (("profile", profile_scenarios), ("disutility", slope_scenarios))}
    settings = {"day": "d1", "n_per_experiment": 5, "profile": {"seed": 1, "bounds": [0.8, 1.2]}, "disutility": {"seed": 42, "bounds": [0.6, 1.4]}, "mean": 1.0, "kappa": 2.0, "selection": "optimistic", "member_optimality_tolerance_DKK": 1e-7, "metrics_tolerance": 1e-5, "scenarios": scenario_data}
    Path("results/q3_step1/scenarios.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("\nStep 1 finished. Results: results/q3_step1/")
    
    # PLOT Q3 PART1 
    plot_Q3_part1_results(profile_results, slope_results, nominal_report)
