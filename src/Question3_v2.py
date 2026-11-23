"""Compare Q3 risk-neutral, mean–CVaR, and chance-constrained pricing.

Run from the project root with src/DP.py and Question3_step1.py in place.
"""
import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.Question3_step1 import (
    part1_metrics,
    save_part1_results,
    scenario_community,
    solve_fixed_price_responses,
)
from src import DP as dp
from src.data_loader import load_community
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



@dataclass(frozen=True)
class RiskSettings:
    confidence: float = 0.90
    cvar_weight: float = 0.50
    violation_probability: float = 0.10
    deficit_limit_DKK: float = 1.0
    ir_loss_limit_DKK: float = 1.0


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
        outside = dp.CommunityModel.I(realized).build()
        with outside.m:
            outside_option = outside.solve()
        outside_seconds += perf_counter() - start
        cases.append({"scenario": scenario, "community": realized, "outside_option": outside_option})
    return cases, {"preparation_seconds": preparation_seconds, "outside_option_seconds": outside_seconds}


def metrics(com, prices, cases, risk=RiskSettings()):
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


def solve_stochastic(com, cases, time_limit=None, verbose=False, *, variant="SDP", risk=RiskSettings()):
    """Use the same stochastic dispatch and settlement model for all variants."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    weights = [case["scenario"]["probability"] for case in cases]
    if not weights or min(weights) < 0 or not np.isclose(sum(weights), 1.0):
        raise ValueError("Scenario probabilities must be nonnegative and sum to one.")
    start = perf_counter()
    with gp.Model(f"Q3_{variant}") as model:
    
        # Logging (file only, no console output)
        model.Params.LogToConsole = 0
        model.Params.LogFile = f"results/q3/solutionlogs/Q3_{variant}_{len(cases)}_v3.log"
        model.Params.OutputFlag = 1

        model.Params.DisplayInterval = 1
    
        # Save incumbent solutions
        model.Params.SolFiles = f"results/q3/solutionlogs/Q3_{variant}_{len(cases)}_v3_incumbent"
    
        # Solver strategy
        model.Params.PreSOS1BigM = 0
    
        # Tolerances
        model.Params.FeasibilityTol = 1e-6
        model.Params.MIPGap = 0.001
    
        # Time limit
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
    raw_community = load_json("params_community_base.json")
    rules = load_json("preference_uncertainty_params.json")
    app, _ = day_data(load_json("appliance_params.json"), load_json("bus_params.json"),
                      load_json("weekly_profiles.json"), "d1")
    spec = rules_to_spec(rules, com.member_ids)
    set_rule(spec, "profile", lo=0.8, hi=1.2, mean=1.0, kappa=2.0)
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
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[5, 10, 12])
    parser.add_argument("--test-size", type=int, default=100, help="Use a smaller value for a smoke test.")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--test-seed", type=int, default=99)
    parser.add_argument("--time-limit", type=float, help="Optional seconds per pricing solve.")
    parser.add_argument("--cvar-confidence", type=float, default=0.90)
    parser.add_argument("--cvar-weight", type=float, default=0.50)
    parser.add_argument("--violation-probability", type=float, default=0.10)
    parser.add_argument("--deficit-limit", type=float, default=1.0, help="Allowed manager deficit in DKK.")
    parser.add_argument("--ir-loss-limit", type=float, default=1.0, help="Allowed loss per member in DKK.")
    parser.add_argument("--weights", nargs="*", type=float, default=[0, 0.25, 0.75])
    parser.add_argument("--probabilities", nargs="*", type=float, default=[0, 0.2, 0.3])
    parser.add_argument("--sensitivity-size", type=int, default=10, help="Training size for sensitivity runs (default: 10).")
    parser.add_argument("--output", type=Path, default=Path("results/q3"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    
    
    # if (min(args.train_sizes) < 2 or args.test_size < 2 or min(args.train_seed, args.test_seed) < 0
    #         or args.train_seed == args.test_seed):
    #     parser.error("Use scenario sizes of at least 2 and distinct nonnegative seeds.")
    if args.time_limit is not None and (not np.isfinite(args.time_limit) or args.time_limit <= 0):
        parser.error("--time-limit must be positive and finite.")
    if not (0 < args.cvar_confidence < 1 and 0 < args.cvar_weight <= 1
            and 0 <= args.violation_probability < 1):
        parser.error("Use a CVaR confidence in (0, 1), a weight in (0, 1], and a violation probability in [0, 1).")
    if any(not np.isfinite(x) or x < 0 for x in (args.deficit_limit, args.ir_loss_limit)):
        parser.error("Settlement limits must be finite and nonnegative.")
    risk = RiskSettings(
        args.cvar_confidence, args.cvar_weight, args.violation_probability,
        args.deficit_limit, args.ir_loss_limit,
    )

    if any(not 0 <= x <= 1 for x in args.weights) or any(not 0 <= x < 1 for x in args.probabilities):
        parser.error("Weights must lie in [0, 1] and probabilities in [0, 1).")
    sensitivity_size = args.sensitivity_size
    if sensitivity_size not in args.train_sizes:
        parser.error("--sensitivity-size must be one of --train-sizes.")
        
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(args.train_sizes))
    com = load_community("base", day="d1")
    start = perf_counter()
    nominal, train, test = make_scenarios(com, max(sizes), args.test_size, args.train_seed, args.test_seed)
    generation_seconds = perf_counter() - start
    write_json(folder / "scenarios.json", {
        "day": "d1", "train_seed": args.train_seed, "test_seed": args.test_seed,
        "train_sizes": sizes, "test_size": args.test_size,
        "profile_rule": {"lo": 0.8, "hi": 1.2, "mean": 1.0, "kappa": 2.0},
        "disutility_slope": "nominal",
        "train_pool": [dict(s, index=[[q, i, k] for (q, i), k in s["index"].items()]) for s in train],
        "test": [dict(s, index=[[q, i, k] for (q, i), k in s["index"].items()]) for s in test],
        "training_weights": "Each prefix of length N is reweighted to 1/N per scenario.",
    })

    nominal_cases, nominal_times = prepare_cases(com, [nominal])
    realized = nominal_cases[0]["community"]
    for base, actual in zip(com.members, realized.members):
        for attr in ("reference_load", "min_daily_energy_kWh", "load_lower_kWh",
                     "load_upper_kWh", "linear_disutility"):
            np.testing.assert_allclose(getattr(actual, attr), getattr(base, attr),
                                       atol=1e-9, rtol=0, err_msg=f"{base.consumer_id} {attr}")

    outside = nominal_cases[0]["outside_option"]["member_cost"]
    start = perf_counter()
    baseline = dp.DynamicPricingModel(
        com, budget_balance=True, ir_baseline=outside, uniform_prices=False,
        time_limit=args.time_limit, mip_gap=1e-6, verbose=args.verbose, name="Q2_DPBBIR_IP",
    ).build()
    with baseline.m:
        build_seconds = perf_counter() - start
        q2 = baseline.solve()
        baseline_info = solver_info(baseline.m, build_seconds)
    diagnostics = {
        "scenario_generation_seconds": generation_seconds, "nominal_preparation": nominal_times,
        "deterministic_solver": baseline_info, "risk_settings": asdict(risk),
    }
    write_json(folder / "diagnostics.json", diagnostics)
    if "price" not in q2:
        raise RuntimeError(f"Q2 produced no incumbent (status {baseline_info['status']}).")
    diagnostics["nominal_variants"] = {}

    for variant in VARIANTS:
        _, report, info = solve_stochastic(com, nominal_cases, args.time_limit, args.verbose,
                                           variant=variant, risk=risk)
        validation = "inconclusive"
        if baseline_info["optimal"] and info["optimal"]:
            difference = abs(info["incumbent_DKK"] - q2["cost"])
            print(f"Nominal objective difference: {difference:.6f} DKK")
            validation = "passed" if difference <= 0.5 else "inconclusive" # relative gao gap is 0.001
        diagnostics["nominal_variants"][variant] = {"solver": info, "validation": validation}
        if variant == "SDP":
            diagnostics.update(nominal_stochastic_solver=info, nominal_validation=validation)
            if report is not None:
                save_part1_results(report, folder, "nominal_stochastic")
        print(f"Nominal Q2 reproduction ({variant}): {validation}")
    write_json(folder / "diagnostics.json", diagnostics)

    train_cases, diagnostics["train_preparation"] = prepare_cases(com, train)
    test_cases, diagnostics["test_preparation"] = prepare_cases(com, test)
    det_test = evaluate_prices(com, q2["price"], test_cases, risk)
    save_part1_results(det_test, folder, "deterministic_test")
    study = {"risk_settings": asdict(risk), "deterministic_test": det_test["summary"]}
    policies = {"deterministic": {i: p.tolist() for i, p in q2["price"].items()}}
    file_tags = {"SDP": "sdp", "CVaR": "cvar", "CVaR+CC": "cvar_cc"}
    print("********************SDTART MODERL TRAINING************************")
    for n in sizes:
        label = f"n{n}"
        prefix = [dict(case, scenario=dict(case["scenario"], probability=1 / n)) for case in train_cases[:n]]
        fits, train_replays, tests = {}, {"DP": evaluate_prices(com, q2["price"], prefix, risk)}, {"DP": det_test}
        run = study[label] = {"variant_solvers": {}, "DP train": train_replays["DP"]["summary"],
                              "DP test": det_test["summary"]}
        save_part1_results(train_replays["DP"], folder, f"{label}_dp_train")
        for name, variant, settings in experiment_matrix(n, sensitivity_size, risk, args.weights, args.probabilities):
            prices, training, info = solve_stochastic(com, prefix, args.time_limit, args.verbose,
                                                       variant=variant, risk=settings)
            run["variant_solvers"][name] = dict(info, risk_settings=asdict(settings))
            if variant == "SDP":
                run["solver"] = info
            state = {GRB.OPTIMAL: "optimal", GRB.INFEASIBLE: "infeasible",
                     GRB.TIME_LIMIT: "time limit"}.get(info["status"], f"status {info['status']}")
            print(f"\nN={n}, {name}, {state}, runtime={info['runtime_seconds']:.2f}s, "
                  f"incumbent={info['incumbent_DKK']}, bound={info['objective_bound_DKK']}, gap={info['mip_gap']}")
            write_json(folder / "study.json", study)
            if prices is None:
                continue
            tag = file_tags.get(name, name.lower())
            fits[name] = training
            train_replays[name] = evaluate_prices(com, prices, prefix, settings)
            tests[name] = evaluate_prices(com, prices, test_cases, settings)
            for phase, report in (("fit", training), ("train", train_replays[name]), ("test", tests[name])):
                save_part1_results(report, folder, f"{label}_{tag}_{phase}")
                run[f"{name} {phase}"] = report["summary"]
            price_key = label if variant == "SDP" else f"{label}_{tag}"
            policies[price_key] = {i: p.tolist() for i, p in prices.items()}
            write_json(folder / "study.json", study)
            write_json(folder / "prices.json", policies)
        if fits:
            print_comparison("Optimized in-sample dispatch", fits)
        print_comparison("In-sample fixed-price replay", train_replays)
        print_comparison("Out-of-sample fixed-price replay", tests)
        write_json(folder / "study.json", study)
        write_json(folder / "diagnostics.json", diagnostics)

    if not any(label != "deterministic" for label in policies):
        raise RuntimeError("The stochastic pricing solves produced no incumbents. See study.json.")


if __name__ == "__main__":
    main()



