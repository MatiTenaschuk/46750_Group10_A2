"""Earlier import path for the fixed-price experiments in src.Question3."""

from pathlib import Path

from src.Question3 import (
    evaluate_scenarios,
    part1_metrics,
    print_part1_results,
    reproduce_question2,
    run_part1,
    save_part1_results,
    scenario_community,
    solve_fixed_price_responses,
    solve_independent_price_responses,
)


if __name__ == "__main__":
    from src.data_loader import load_community

    com = load_community("base", day="d1")
    run_part1(com, Path("results/q3_step1"))
