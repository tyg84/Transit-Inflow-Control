# Equity-Oriented Rail Inflow Control

This repository contains the Python implementation used in the manuscript
"Equity-Oriented Inflow Control in Congested Urban Rail Systems."

The code implements an event-based transit simulation, the proposed capacity
reallocation algorithm, the existing MLR, Bayesian optimization (BO), and
differential evolution (DE) benchmarks, network-size experiments, and the
scripts used to calculate and visualize the reported results.

## Code map

- `A01`--`A06`: synthetic input generation and benchmark-case preparation.
- `B01`: passenger-train discrete-event simulation.
- `B03`: proposed event-based capacity reallocation algorithm.
- `B04`: Bayesian optimization benchmark (the historical filename uses `BYO`).
- `B05`: differential evolution benchmark.
- `B06`: existing MLR benchmark.
- `C00`--`C04`: convergence, distribution, and spatial figures.
- `D01`--`D02`: network-size experiments and existing benchmarks.
- `E01`: passenger-level equity-efficiency analysis.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run scripts from the repository root because paths are resolved relative to the
current working directory. The experiment scripts expect case-specific input
files under `data/<case_name>/` and write iteration logs under
`output/<case_name>/`.

## Synthetic input generation

The small files `data/testSubwayStation.csv`, `data/headway.csv`, and
`data/train_capacity.csv` specify the synthetic system patterned after the
Shanghai Metro. Generate the reference case in this order:

```bash
python A01_generate_basic_data.py
python A02_generate_paths.py
python A03_generate_events.py
python A04_generate_demand_data.py
python A05_generate_train_capacity.py
python A06_prepare_benchmark_cases.py
```

These scripts create the synthetic network, passenger paths, train events,
individual demand, and adjusted train capacity under `data/reference/`. Fixed
random seeds in `A04_generate_demand_data.py` make the reference demand
reproducible. `A06_prepare_benchmark_cases.py` copies those same inputs to the
existing MLR, BO, and DE case directories; it does not alter any benchmark
method. `D01_network_size_experiment.py` generates the two-, three-, and
four-line synthetic cases used in the network-size experiment.

## Reference experiment

The published code defaults to these case names:

```text
Proposed: reference
MLR:      equity_efficiency_MLR
BO:       BYO
DE:       DE
```

The benchmark algorithms are unchanged from those used in the manuscript. The
method-specific folders allow all methods to use the same reference demand,
timetable, capacity, and passenger paths while retaining separate output logs.

Typical execution order:

```bash
python B03_control_strategies.py
python B06_rule_based_using_current_max_LB.py
python B04_black_box_optimization_BYO.py
python B05_black_box_optimization_DE.py
python C01_plot_convergence_comparison.py
python E01_equity_efficiency_tradeoff.py
```

Generated case directories and full simulation logs are not committed because
of their size. They can be regenerated with the scripts above; the detailed
logs underlying the manuscript results are available from the corresponding
author upon reasonable request.
