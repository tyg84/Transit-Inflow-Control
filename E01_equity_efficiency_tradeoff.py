"""Reproduce the equity--efficiency comparison on one common demand scenario.

The script replays stored controls instead of changing any solution method. It
computes passenger-level waiting metrics for the uncontrolled case and the
selected Proposed, MLR, BO, and DE solutions. It also evaluates every Proposed
iteration to expose the trade-off between the min--max objective and total
passenger waiting time.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import B03_control_strategies as control
from B01_simulation import (
    assign_passenger_path,
    generate_event_list,
    process_passenger_group_by_origin,
)


SIMULATION_START_TIMESTAMP = 6 * 3600
SIMULATION_END_TIMESTAMP = 9 * 3600
METHOD_ORDER = ["Uncontrolled", "Proposed", "MLR", "BO", "DE"]
CONTROL_PATTERN = re.compile(
    r"control_board_numbers:\s*([^,]+),\s*control_factor:\s*(.+)$"
)


def empty_logs():
    return {
        "trajectory_log": {
            "passenger_id": [],
            "trajectory_type": [],
            "trajectory_time": [],
            "trajectory_platform": [],
        },
        "train_load_log": [],
        "platform_queue_log": [],
        "left_behind_log": [],
        "train_boarding_log": {},
        "left_behind_log_temp": {},
        "completed_passenger_objects": {},
    }


def parse_control_log(train_log_path, mode):
    train_log = pd.read_csv(train_log_path, dtype={"train_id": str})
    departures = train_log.loc[train_log["train_load_type"].eq("Departure")]
    board_controls = {}
    factor_controls = {}

    for row in departures.itertuples(index=False):
        match = CONTROL_PATTERN.fullmatch(str(row.control_log).strip())
        if match is None:
            raise ValueError(f"Unrecognized control log in {train_log_path}: {row.control_log}")
        board_raw, factor_raw = (value.strip() for value in match.groups())
        key = (str(row.train_id), str(row.platform_id))
        if board_raw != "None":
            board_controls[key] = {"Control": int(float(board_raw)), "Onboard": None}
        if factor_raw != "None":
            factor_controls[key] = float(factor_raw)

    if mode == "board":
        return board_controls, None
    if mode == "factor":
        return {}, factor_controls
    raise ValueError(f"Unknown control mode: {mode}")


class ReplayData:
    def __init__(self, case_name):
        data_dir = Path("data") / case_name
        demand = pd.read_csv(data_dir / "individual_demands.csv")
        self.demand = demand.loc[
            (demand["tap_in_timestamp"] > SIMULATION_START_TIMESTAMP)
            & (demand["tap_in_timestamp"] < SIMULATION_END_TIMESTAMP)
        ].copy()

        paths = pd.read_csv(data_dir / "paths.csv")
        control.pax_path_dict, passenger_paths = assign_passenger_path(self.demand, paths)
        self.passenger_paths = passenger_paths
        self.grouped_template = process_passenger_group_by_origin(passenger_paths)

        events = pd.read_csv(data_dir / "events.csv", dtype={"train_id": str})
        events = events.loc[
            (events["event_timestamp"] > SIMULATION_START_TIMESTAMP)
            & (events["event_timestamp"] < SIMULATION_END_TIMESTAMP)
        ].copy()
        self.event_list = generate_event_list(events)

        train_capacity = pd.read_csv(data_dir / "train_capacity_adjusted.csv")
        control.train_capacity_dict = train_capacity.set_index(
            ["line_id", "direction_id"]
        )["train_capacity"].to_dict()
        control.SIMULATION_START_TIMESTAMP = SIMULATION_START_TIMESTAMP

    def replay(self, train_log_path, mode):
        board_controls, factor_controls = parse_control_log(train_log_path, mode)
        logs = empty_logs()
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            (
                _,
                _,
                passenger_objects,
                all_platforms,
                all_trains,
                logs,
                _,
            ) = control.simulation_with_control(
                self.event_list,
                {},
                {},
                {},
                logs,
                0,
                board_controls,
                factor_controls,
                copy.deepcopy(self.grouped_template),
                BOARD_NUM_CONTROL=(mode == "board"),
            )
        return (
            passenger_objects,
            all_platforms,
            all_trains,
            logs,
            board_controls,
            factor_controls,
        )


def latest_passenger_states(passenger_objects, all_platforms, all_trains, logs):
    states = dict(passenger_objects)

    def update(passenger):
        passenger_id = int(passenger.passenger_id)
        current = states.get(passenger_id)
        if current is None or len(passenger.trajectory["trajectory_type"]) > len(
            current.trajectory["trajectory_type"]
        ):
            states[passenger_id] = passenger

    for platform in all_platforms.values():
        for passenger in platform.passenger_list:
            update(passenger)
    for train in all_trains.values():
        for passenger in train.passenger_list:
            update(passenger)
    for passenger in logs["completed_passenger_objects"].values():
        update(passenger)
    return states


def passenger_waiting_table(replay_data, passenger_states, completed_ids):
    rows = []
    for demand_row in replay_data.demand.itertuples(index=False):
        passenger_id = int(demand_row.passenger_id)
        tap_in = float(demand_row.tap_in_timestamp)
        passenger = passenger_states.get(passenger_id)

        if passenger is None:
            total_wait = max(0.0, SIMULATION_END_TIMESTAMP - tap_in)
            rows.append(
                {
                    "passenger_id": passenger_id,
                    "origin_station_id": int(demand_row.origin_station_id),
                    "total_wait_seconds": total_wait,
                    "completed_trip": False,
                    "max_left_behind": 0,
                }
            )
            continue

        waiting_start = tap_in
        total_wait = 0.0
        for trajectory_type, trajectory_time in zip(
            passenger.trajectory["trajectory_type"],
            passenger.trajectory["trajectory_time"],
        ):
            trajectory_time = float(trajectory_time)
            if trajectory_type == "Transfer":
                waiting_start = trajectory_time
            elif trajectory_type == "Boarding":
                if waiting_start is not None:
                    total_wait += max(0.0, trajectory_time - waiting_start)
                waiting_start = None

        if waiting_start is not None:
            total_wait += max(0.0, SIMULATION_END_TIMESTAMP - waiting_start)

        rows.append(
            {
                "passenger_id": passenger_id,
                "origin_station_id": int(demand_row.origin_station_id),
                "total_wait_seconds": total_wait,
                "completed_trip": passenger_id in completed_ids,
                "max_left_behind": max(passenger.left_behind_times.values(), default=0),
            }
        )

    return pd.DataFrame(rows)


def objective_value(logs):
    if not logs["left_behind_log"]:
        return 0
    return int(max(row["left_behind_times"] for row in logs["left_behind_log"]))


def records_at_objective(logs, w_max):
    return int(
        sum(row["left_behind_times"] == w_max for row in logs["left_behind_log"])
    )


def boarding_counts(logs):
    if not logs["left_behind_log"]:
        return {}
    frame = pd.DataFrame(logs["left_behind_log"])
    counts = frame.groupby(["boarded_train_id", "boarding_platform"]).size()
    return {(str(train), str(platform)): int(value) for (train, platform), value in counts.items()}


def restricted_events(board_controls, factor_controls, baseline_counts, method_counts):
    candidate_events = set(board_controls)
    candidate_events.update(
        key for key, factor in (factor_controls or {}).items() if factor < 1.0
    )
    return {
        key
        for key in candidate_events
        if method_counts.get(key, 0) < baseline_counts.get(key, 0)
    }


def affected_passengers(passenger_states, restricted_event_set):
    affected = set()
    for passenger_id, passenger in passenger_states.items():
        if any(
            (str(train_id), str(platform_id)) in restricted_event_set
            for platform_id, train_ids in passenger.passed_train_id.items()
            for train_id in train_ids
        ):
            affected.add(int(passenger_id))
    return affected


def summarize_method(
    method,
    iteration,
    waiting,
    w_max,
    baseline_waiting,
    affected_ids,
):
    merged = waiting.merge(
        baseline_waiting[["passenger_id", "total_wait_seconds"]],
        on="passenger_id",
        how="left",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    difference = merged["total_wait_seconds"] - merged["total_wait_seconds_baseline"]
    tolerance = 1e-9
    improved = int((difference < -tolerance).sum())
    worsened = int((difference > tolerance).sum())

    affected_mask = merged["passenger_id"].isin(affected_ids)
    if affected_mask.any():
        upstream_added = np.maximum(difference.loc[affected_mask], 0.0).mean() / 60.0
    else:
        upstream_added = 0.0

    waiting_minutes = waiting["total_wait_seconds"] / 60.0
    throughput = int(waiting["completed_trip"].sum())
    return {
        "method": method,
        "iteration": int(iteration),
        "W_Max": int(w_max),
        "max_wait_min": float(waiting_minutes.max()),
        "mean_wait_min": float(waiting_minutes.mean()),
        "p95_wait_min": float(waiting_minutes.quantile(0.95)),
        "total_wait_passenger_hours": float(waiting["total_wait_seconds"].sum() / 3600.0),
        "passengers_improved": improved,
        "passengers_worsened": worsened,
        "upstream_passengers_affected": int(len(affected_ids)),
        "upstream_additional_delay_min_per_affected_passenger": float(upstream_added),
        "passenger_throughput": throughput,
        "passenger_throughput_pct": float(100.0 * throughput / len(waiting)),
    }


def replay_and_measure(replay_data, train_log_path, mode, baseline_counts):
    (
        passengers,
        platforms,
        trains,
        logs,
        board_controls,
        factor_controls,
    ) = replay_data.replay(train_log_path, mode)
    passenger_states = latest_passenger_states(passengers, platforms, trains, logs)
    completed_ids = set(logs["completed_passenger_objects"])
    waiting = passenger_waiting_table(replay_data, passenger_states, completed_ids)
    method_counts = boarding_counts(logs)
    restricted = restricted_events(
        board_controls, factor_controls, baseline_counts, method_counts
    )
    affected = affected_passengers(passenger_states, restricted)
    w_max = objective_value(logs)
    return {
        "waiting": waiting,
        "W_Max": w_max,
        "records_at_W_Max": records_at_objective(logs, w_max),
        "boarding_counts": method_counts,
        "affected": affected,
        "restricted_events": restricted,
    }


def find_best_iteration(output_case, maximum_iterations):
    values = []
    for iteration in range(maximum_iterations):
        path = Path("output") / output_case / f"left_behind_log_iteration_{iteration}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["left_behind_times"])
        w_max = int(frame["left_behind_times"].max())
        records_at_w_max = int(frame["left_behind_times"].eq(w_max).sum())
        values.append((w_max, records_at_w_max, iteration))
    if not values:
        raise FileNotFoundError(f"No left-behind logs found for {output_case}")
    return min(values)[2]


def plot_tradeoff(history, first_best_iteration, selected_iteration, output_dir):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    history = history.sort_values("iteration").reset_index(drop=True)
    x = history["total_wait_passenger_hours"]
    y = history["W_Max"]
    first_best = history.loc[history["iteration"].eq(first_best_iteration)].iloc[0]
    selected = history.loc[history["iteration"].eq(selected_iteration)].iloc[0]
    baseline = history.loc[history["iteration"].eq(0)].iloc[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.scatter(
        x,
        y,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.45,
        s=34,
        label="Proposed iterations",
        zorder=2,
    )
    ax.scatter(
        baseline["total_wait_passenger_hours"],
        baseline["W_Max"],
        marker="D",
        color="black",
        s=62,
        label="Uncontrolled",
        zorder=4,
    )
    ax.scatter(
        first_best["total_wait_passenger_hours"],
        first_best["W_Max"],
        marker="s",
        facecolor="white",
        edgecolor="#1f77b4",
        linewidth=1.4,
        s=82,
        label=f"First min-max solution (Iter. {first_best_iteration})",
        zorder=4,
    )
    ax.scatter(
        selected["total_wait_passenger_hours"],
        selected["W_Max"],
        marker="*",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.55,
        s=180,
        label=f"Equity-refined solution (Iter. {selected_iteration})",
        zorder=5,
    )

    record = np.inf
    for row in history.itertuples(index=False):
        if row.W_Max < record:
            record = row.W_Max
            ax.annotate(
                f"Iter. {int(row.iteration)}",
                (row.total_wait_passenger_hours, row.W_Max),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=9.5,
                color="black" if row.iteration == 0 else "#1f4e79",
            )

    ax.set_xlabel("Total passenger waiting time (passenger-hours)")
    ax.set_ylabel(r"Maximum left-behind time, $W_{\mathrm{Max}}$")
    ax.set_yticks(np.arange(int(y.min()), int(y.max()) + 1))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.legend(loc="upper right")

    output_base = output_dir / "equity_efficiency_tradeoff"
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="reference")
    parser.add_argument("--proposed-case", default="reference")
    parser.add_argument("--mlr-case", default="equity_efficiency_MLR")
    parser.add_argument("--bo-case", default="BYO")
    parser.add_argument("--de-case", default="DE")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--proposed-iteration", type=int)
    parser.add_argument("--mlr-iteration", type=int)
    parser.add_argument("--bo-iteration", type=int)
    parser.add_argument("--de-iteration", type=int)
    parser.add_argument("--output", default="output/equity_efficiency")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_data = ReplayData(args.case)

    baseline_path = (
        Path("output") / args.proposed_case / "train_log_iteration_0.csv"
    )
    baseline = replay_and_measure(replay_data, baseline_path, "board", {})
    baseline_counts = baseline["boarding_counts"]
    baseline_waiting = baseline["waiting"]

    history_path = output_dir / "proposed_tradeoff_source_data.csv"
    if args.proposed_iteration is not None:
        history = pd.read_csv(history_path)
        minimum_w_max = history["W_Max"].min()
        proposed_first_best_iteration = int(
            history.loc[history["W_Max"].eq(minimum_w_max), "iteration"].min()
        )
        proposed_selected_iteration = args.proposed_iteration
        selected_path = (
            Path("output")
            / args.proposed_case
            / f"train_log_iteration_{proposed_selected_iteration}.csv"
        )
        proposed_selected = replay_and_measure(
            replay_data, selected_path, "board", baseline_counts
        )
        if proposed_selected["W_Max"] != minimum_w_max:
            raise ValueError("The requested Proposed iteration does not attain minimum W_Max")
    else:
        proposed_history = []
        proposed_first_best_iteration = None
        proposed_selected_iteration = None
        proposed_selected = None
        proposed_selection_key = None
        minimum_w_max = np.inf
        for iteration in range(args.iterations):
            print(f"Replay Proposed iteration {iteration}/{args.iterations - 1}", flush=True)
            if iteration == 0:
                result = baseline
            else:
                path = (
                    Path("output")
                    / args.proposed_case
                    / f"train_log_iteration_{iteration}.csv"
                )
                result = replay_and_measure(replay_data, path, "board", baseline_counts)

            total_wait = result["waiting"]["total_wait_seconds"].sum() / 3600.0
            proposed_history.append(
                {
                    "iteration": iteration,
                    "W_Max": result["W_Max"],
                    "records_at_W_Max": result["records_at_W_Max"],
                    "total_wait_passenger_hours": total_wait,
                    "mean_wait_min": result["waiting"]["total_wait_seconds"].mean()
                    / 60.0,
                    "passenger_throughput": int(result["waiting"]["completed_trip"].sum()),
                }
            )
            if result["W_Max"] < minimum_w_max:
                minimum_w_max = result["W_Max"]
                proposed_first_best_iteration = iteration
            selection_key = (
                result["W_Max"],
                result["records_at_W_Max"],
                iteration,
            )
            if proposed_selection_key is None or selection_key < proposed_selection_key:
                proposed_selection_key = selection_key
                proposed_selected = result
                proposed_selected_iteration = iteration

        history = pd.DataFrame(proposed_history)
        history.to_csv(history_path, index=False)

    plot_tradeoff(
        history,
        proposed_first_best_iteration,
        proposed_selected_iteration,
        output_dir,
    )

    mlr_iteration = args.mlr_iteration
    if mlr_iteration is None:
        mlr_iteration = find_best_iteration(args.mlr_case, args.iterations)
    bo_iteration = args.bo_iteration
    if bo_iteration is None:
        bo_iteration = find_best_iteration(args.bo_case, args.iterations)
    de_iteration = args.de_iteration
    if de_iteration is None:
        de_iteration = find_best_iteration(args.de_case, args.iterations)
    selected_specs = {
        "MLR": (args.mlr_case, mlr_iteration, "board"),
        "BO": (args.bo_case, bo_iteration, "factor"),
        "DE": (args.de_case, de_iteration, "factor"),
    }
    selected_results = {
        "Uncontrolled": (0, baseline),
        "Proposed": (proposed_selected_iteration, proposed_selected),
    }
    for method, (case_name, iteration, mode) in selected_specs.items():
        print(f"Replay {method} iteration {iteration}", flush=True)
        path = Path("output") / case_name / f"train_log_iteration_{iteration}.csv"
        selected_results[method] = (
            iteration,
            replay_and_measure(replay_data, path, mode, baseline_counts),
        )

    summary_rows = []
    passenger_frames = []
    for method in METHOD_ORDER:
        iteration, result = selected_results[method]
        summary_rows.append(
            summarize_method(
                method,
                iteration,
                result["waiting"],
                result["W_Max"],
                baseline_waiting,
                result["affected"],
            )
        )
        passenger_frame = result["waiting"].copy()
        passenger_frame.insert(0, "method", method)
        passenger_frame["affected_by_restriction"] = passenger_frame[
            "passenger_id"
        ].isin(result["affected"])
        passenger_frames.append(passenger_frame)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "equity_efficiency_summary.csv", index=False)
    pd.concat(passenger_frames, ignore_index=True).to_csv(
        output_dir / "passenger_waiting_source_data.csv", index=False
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
