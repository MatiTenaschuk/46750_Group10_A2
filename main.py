"""Run one question of the energy community assignment.

    python main.py
    python main.py q1 --sensitivity --settlement
    python main.py q2 --benchmark I --time-limit 60
    python main.py q3 --part 1
    python main.py q3 --part 2
    python main.py q3
"""
from __future__ import annotations

import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 argument_default=argparse.SUPPRESS)
    ap.add_argument("question", nargs="?", default=None, choices=("q1", "q2", "q3"))
    ap.add_argument("--community", default="base", help="community name from data/params_community_<name>.json")
    ap.add_argument("--day", default="d1", help="day of the week, d1 to d7 (default: d1)")
    ap.add_argument("--members", action="store_true", help="show each member's input data")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--time-limit", type=float,
                    help="seconds per pricing solve (Q2 defaults to 15, Q3 has no default limit)")

    q1 = ap.add_argument_group("Question 1")
    q1.add_argument("--sensitivity", action="store_true", help="vary the discount and the capacity limit")
    q1.add_argument("--settlement", action="store_true", help="calculate member settlements at the internal price")

    q2 = ap.add_argument_group("Question 2")
    q2.add_argument("--method", choices=("bigM", "SOS1"), help="complementarity formulation (default: bigM)")
    q2.add_argument("--benchmark", choices=("I", "R"), help="individual rationality benchmark (default: I)")
    q2.add_argument("--skip-indifference", action="store_true", help="skip the member indifference analysis")

    q3 = ap.add_argument_group("Question 3")
    q3.add_argument("--part", choices=("1", "2", "all"), help="fixed prices (1), stochastic pricing (2), or both (default)")
    q3.add_argument("--part1-draws", type=int, help="draws per Part 1 experiment (default: 5, report: 15)")
    q3.add_argument("--train-sizes", type=int, nargs="+", help="nested training sizes (default: 5 10 12)")
    q3.add_argument("--test-size", type=int, help="number of shared test scenarios (default: 100)")
    q3.add_argument("--train-seed", type=int, help="seed for training draws (default: 1)")
    q3.add_argument("--test-seed", type=int, help="seed for test draws (default: 99)")
    q3.add_argument("--cvar-confidence", type=float, help="CVaR confidence level (default: 0.90)")
    q3.add_argument("--cvar-weight", type=float, help="CVaR weight in the reference fit (default: 0.50)")
    q3.add_argument("--violation-probability", type=float, help="joint violation allowance (default: 0.10)")
    q3.add_argument("--deficit-limit", type=float, help="allowed manager deficit in DKK (default: 1)")
    q3.add_argument("--ir-loss-limit", type=float, help="allowed loss per member in DKK (default: 1)")
    q3.add_argument("--weights", type=float, nargs="*", help="additional CVaR weights (default: 0 0.25 0.75)")
    q3.add_argument("--probabilities", type=float, nargs="*", help="additional violation allowances (default: 0 0.2 0.3)")
    q3.add_argument("--sensitivity-size", type=int, help="training size for both parameter sweeps (default: 10)")
    q3.add_argument("--output", type=Path, help="Q3 output directory (default: results/q3)")
    q3.add_argument("--verbose", action="store_true", help="show Gurobi pricing logs in the terminal")

    # Pass only explicitly supplied experiment options to the question's run function.
    options = vars(ap.parse_args(argv))
    question = options.pop("question")
    day = options.pop("day")
    community = options.pop("community")
    members = options.pop("members", False)
    demo = options.pop("demo", False)
    q1_only = {"sensitivity", "settlement"}
    q2_only = {"method", "benchmark", "skip_indifference"}
    if question == "q3":
        invalid = options.keys() & (q1_only | q2_only)
    else:
        allowed = {"q1": q1_only, "q2": q2_only | {"time_limit"}}.get(question, set())
        invalid = options.keys() - allowed
    if invalid:
        flags = ", ".join("--" + key.replace("_", "-") for key in sorted(invalid))
        ap.error(f"{flags} cannot be used with {question or 'the data summary'}")

    from src.data_loader import load_community

    com = load_community(community, day=day)
    print(com.summary(), "\n")
    if members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / community).mkdir(parents=True, exist_ok=True)
    if question == "q1":
        from src.Question1 import run

        run(com, RESULTS_DIR / "q1", **options)
    elif question == "q2":
        from src.Question2 import run

        run(com, RESULTS_DIR / "q2", day=day, **options)
    elif question == "q3":
        from src.Question3 import run

        out = options.pop("output", RESULTS_DIR / "q3")
        run(com, out, **options)

    if demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
