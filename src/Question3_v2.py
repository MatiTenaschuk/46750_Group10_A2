"""Earlier import path and command for stochastic pricing in src.Question3."""

from src.Question3 import (
    ACCOUNTING_TOL,
    METRICS_TOL,
    VARIANTS,
    RiskSettings,
    ScenarioBlock,
    evaluate_prices,
    experiment_matrix,
    make_scenarios,
    metrics,
    part1_metrics,
    prepare_cases,
    print_comparison,
    save_part1_results,
    scenario_community,
    solve_fixed_price_responses,
    solve_risk_neutral,
    solve_stochastic,
    solver_info,
    write_json,
)


def main(argv=None):
    import sys

    from main import main as run_question

    args = sys.argv[1:] if argv is None else argv
    run_question(["q3", "--part", "2", *args])


if __name__ == "__main__":
    main()
