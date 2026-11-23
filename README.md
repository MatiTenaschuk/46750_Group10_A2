# Assignment 2 · An energy community as a flexibility provider

We study 16 electricity consumers over one day. Each member can shift part of their demand. Some members also have solar panels, a battery, or both. We first compare individual and community contracts. We then find the prices the manager can offer and test how those prices perform when the members' preferences change.

## Getting started

Open a terminal in the project folder, where you can see `main.py` and `data/`. Use your assignment Python environment if you already have one. For a fresh setup, use Python 3.11 and install the packages below. The full optimization models need a Gurobi licence that supports their size. The saved Q3 runs used Gurobi 13.0.3.

```bash
python -m pip install numpy pandas matplotlib gurobipy==13.0.3
python main.py
```

The second command reads the data and prints a summary. For the base case, you should see 16 members and 24 hourly capacity limits. The default day is `d1`. To see another day, add `--day d4`. Use `--members` to see each member's data, or `--demo` to see how scenarios are drawn.

## Commands for the assignment

| What you want to run | Command |
|---|---|
| Q1 · Compare the three contracts | `python main.py q1` |
| Q1 · Include the contract sensitivities and member settlements | `python main.py q1 --sensitivity --settlement` |
| Q2 · Run the pricing experiments | `python main.py q2` |
| Q3 · Run both parts with the repository defaults | `python main.py q3` |
| Q3 Part 1 · Test the deterministic prices with the five-draw default | `python main.py q3 --part 1` |
| Q3 Part 1 · Use the 15 draws per experiment described in the report | `python main.py q3 --part 1 --part1-draws 15` |
| Q3 Part 2 · Train and test the stochastic prices | `python main.py q3 --part 2` |

To run both parts of Q3 with the report's Part 1 sample size, use `python main.py q3 --part1-draws 15`. Each question can also run on its own. The code solves any earlier model it needs. To see all options, run `python main.py --help`.

### Question 1 · Is it worth forming a community?

We compare regular tariffs (**R**), individual capacity contracts (**I**) and one shared contract with central scheduling (**C**). The difference between the costs of I and C gives the saving from pooling the contracts. The sensitivity study varies the tariff discount and the capacity limit. The settlement study shows what each member pays and how much the manager has left after paying the bill.

The main comparison is printed in the terminal. The import figure, sensitivity tables and settlement results go in `results/q1/`, depending on the options selected.

### Question 2 · Which prices should the manager offer?

The manager posts hourly prices and each member chooses the cheapest schedule for their own needs. We first use a common price, then add budget balance and individual rationality. In the last experiment, each member can receive a different price. Budget balance means that the money collected covers the community bill exactly. Individual rationality means that each member's cost is at most what they would pay under their outside option.

The default outside option is I. The default solution method is `bigM`, with 15 seconds per pricing solve. For example, `python main.py q2 --benchmark R --method SOS1 --time-limit 60` uses R as the outside option and gives the alternative formulation one minute per solve. The terminal shows the comparisons. Hourly results and member tables for successful solves go in `results/q2/`.

### Question 3 · How well do prices work under uncertainty?

**Part 1.** We keep the deterministic Q2 prices fixed. We change the preferred load profile in five scenarios (seed 2) and the cost of moving away from that profile in five separate scenarios (seed 3). For the report we use 15 draws per experiment with the same seeds. Both scaling parameters range from 0.8 to 1.2. If several schedules are equally good for the members, we select the combination with the lowest community cost. The tables are in `results/q3/part1/`. The file `nominal.json` records the exact prices used. When there are several optimal prices, a new solve can give the same nominal cost and different results on the draws. The overview and member figures are in its `figures/` folder.

**Part 2.** We vary the load profiles and choose prices using the training scenarios. SDP minimizes average community cost. CVaR gives equal weight to average cost and the average in the most expensive 10% of outcomes. CVaR+CC also limits the probability that the manager's deficit or any member's loss exceeds 1 DKK. Its reference allowance is 10%. All three models enforce budget balance and each member’s individual rationality on average over the training draws.

| Experiment | Default settings |
|---|---|
| Training size | 5, 10 and 12 draws from the same pool, seed 1. The larger sets include the smaller ones. |
| Testing | The same 100 unseen draws for all policies, seed 99. The deterministic Q2 prices provide the benchmark. |
| Weight on CVaR | 0, 0.25, 0.50 and 0.75 at 10 scenarios. The attempted weight of 1 is excluded from the default run. |
| Allowed probability of a settlement violation | 0, 0.10, 0.20 and 0.30 at 10 scenarios, with the CVaR weight fixed at 0.50. |

The full run can take hours. To run just the three main models at 12 scenarios, the two empty lists below switch off the extra sweeps. A separate output folder keeps the full study available.

```bash
python main.py q3 --part 2 --train-sizes 12 --weights --probabilities --output results/q3_n12
```

**Reading the results.** Start with `results/q3/study.json` and look under `n5`, `n10` or `n12`. A **fit** uses the schedules chosen during training. A **train** replay fixes those prices and solves the members' responses again on the training draws. A **test** replay does the same on unseen draws. For example, `CVaR+CC test` under `n12` shows how the 12-scenario policy performed on the 100 test draws.

For a member-level example, open `n12_cvar_cc_test_members.csv` in a spreadsheet and filter the `member` column to `C1`. A positive `IR_gap_DKK` means C1 was worse off than under their individual contract in that draw. The chosen prices are in `prices.json`, the draws in `scenarios.json`, the nominal checks in `diagnostics.json`, and the solver logs in `solutionlogs/`. Results are saved as the experiments finish. The console tables appear after all runs at each training size.

## Where to find the code

`main.py` reads your command. `src/data_loader.py` reads the community, appliances, weekly profiles and contract from `data/`. The experiments themselves are in `src/Question1.py`, `src/Question2.py` and `src/Question3.py`. Q3 uses `src/DP.py` for the member and pricing models and `src/scenario_generation.py` to draw preferences. The figures for Q1 and Q3 Part 1 come from `src/plotting.py`.
