"""Load the energy community and run an Assignment 2 question.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
    python main.py q1                                # solve R, I and C
    python main.py q1 --sensitivity --settlement     # include both Question 1 analyses
    python main.py q2                                # pricing experiments, using I for individual rationality
    python main.py q2 --benchmark R --time-limit 60  # use R and allow 60 seconds per pricing solve
    
    
    ******************* Q2 IS A BIT SLOW IN THE EXPERIMENTS WITH INFEASIBILTY *************************
    ******************* IF YOU WANT REDUCE THE TIME LIMIT *********************************************
"""
from __future__ import annotations

import argparse
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", choices=("q1", "q2"), help="assignment question to run")
    ap.add_argument("--community", default="base", help="base (shipped) or your own params_community_<name>.json")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--sensitivity", action="store_true", help="include the Question 1 contract sensitivity study")
    ap.add_argument("--settlement", action="store_true", help="include the Question 1 dual-price settlement analysis")
    q2 = ap.add_argument_group("Question 2 options")
    q2.add_argument("--method", choices=("bigM", "SOS1"), default=argparse.SUPPRESS,
                    help="complementarity formulation (default: bigM)")
    q2.add_argument("--time-limit", type=float, default=argparse.SUPPRESS,
                    help="time limit in seconds per pricing solve (default: 15)")
    q2.add_argument("--benchmark", choices=("I", "R"), default=argparse.SUPPRESS,
                    help="individual rationality benchmark (default: I, individual contracts)")
    q2.add_argument("--skip-indifference", action="store_true", default=argparse.SUPPRESS,
                    help="skip the optimal-response indifference analysis")
    args = ap.parse_args(argv)
    if args.question != "q1" and (args.sensitivity or args.settlement):
        ap.error("--sensitivity and --settlement require q1")
    q2_options = {key: getattr(args, key) for key in ("method", "time_limit", "benchmark", "skip_indifference")
                  if hasattr(args, key)}
    if args.question != "q2" and q2_options:
        ap.error("--method, --time-limit, --benchmark and --skip-indifference require q2")

    from src.data_loader import load_community

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    if args.question == "q1":
        from src.Question1 import run

        run(com, RESULTS_DIR / "q1", sensitivity=args.sensitivity, settlement=args.settlement)
    elif args.question == "q2":
        from src.Question2 import run

        run(com, RESULTS_DIR / "q2", day=args.day, **q2_options)

    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
