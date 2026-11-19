"""Compatibility imports and module entry point for Question 2."""

from src.Question2 import (
    CommunityModel,
    PricingModel,
    add_member,
    check_C_prices,
    indifferent_member_hours,
    member_alone,
    member_table,
    run,
    set_start,
    solve_experiment,
    summarize_indifference,
    validate_dp,
)

__all__ = [
    "CommunityModel",
    "PricingModel",
    "add_member",
    "check_C_prices",
    "indifferent_member_hours",
    "member_alone",
    "member_table",
    "run",
    "set_start",
    "solve_experiment",
    "summarize_indifference",
    "validate_dp",
]


if __name__ == "__main__":
    import sys

    from main import main

    main(["q2", *sys.argv[1:]])
