import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import B03_control_strategies as control
from B01_simulation import (
    assign_passenger_path,
    generate_event_list,
    process_passenger_group_by_origin,
)
from D01_network_size_experiment import (
    MAX_ITER,
    NETWORK_CASES,
    SIMULATION_END_TIMESTAMP,
    SIMULATION_START_TIMESTAMP,
    run_mlr_control,
    summarize_best_max_lb,
)


class EvaluationLimitReached(Exception):
    pass


def select_platform_train_id(data_case_name, events):
    lb_log_original = pd.read_csv(
        f"output/{data_case_name}/left_behind_log_iteration_0.csv"
    )
    all_lb_pax = lb_log_original.loc[
        lb_log_original["left_behind_times"] > 0
    ].copy()
    all_lb_platform = all_lb_pax[["boarding_platform"]].drop_duplicates()
    min_time = np.min(all_lb_pax["arrival_time_at_platform"])
    max_time = np.max(all_lb_pax["arrival_time_at_platform"])
    all_lb_platform["boarded_line_id"] = all_lb_platform[
        "boarding_platform"
    ].apply(lambda x: int(x.split("_")[1]))
    all_lb_platform["boarded_direction_id"] = all_lb_platform[
        "boarding_platform"
    ].apply(lambda x: int(x.split("_")[2]))

    all_platforms = pd.read_csv(f"data/{data_case_name}/platforms.csv")
    all_lb_platform = all_lb_platform.merge(
        all_platforms[["platform_id", "stop_seq"]],
        left_on=["boarding_platform"],
        right_on=["platform_id"],
    )
    all_lb_platform = all_lb_platform.rename(
        columns={"stop_seq": "board_stop_seq"}
    ).drop(columns=["platform_id"])
    all_used_platform = all_platforms.merge(all_lb_platform, how="cross")
    all_used_platform = all_used_platform.loc[
        (all_used_platform["line_id"] == all_used_platform["boarded_line_id"])
        & (
            all_used_platform["direction_id"]
            == all_used_platform["boarded_direction_id"]
        )
        & (all_used_platform["stop_seq"] <= all_used_platform["board_stop_seq"])
    ]
    all_used_platform = all_used_platform[["platform_id"]].drop_duplicates()

    used_events = events.loc[
        (events["event_timestamp"] <= max_time)
        & (events["event_timestamp"] >= min_time + 20 * 60)
    ]
    used_events = used_events.merge(all_used_platform, on=["platform_id"])
    return used_events


def load_simulation_context(data_case_name):
    control.SIMULATION_START_TIMESTAMP = SIMULATION_START_TIMESTAMP

    train_capacity_df = pd.read_csv(f"data/{data_case_name}/train_capacity_adjusted.csv")
    control.train_capacity_dict = train_capacity_df.set_index(
        ["line_id", "direction_id"]
    )["train_capacity"].to_dict()

    passenger_df = pd.read_csv(f"data/{data_case_name}/individual_demands.csv")
    passenger_df = passenger_df.loc[
        (passenger_df["tap_in_timestamp"] > SIMULATION_START_TIMESTAMP)
        & (passenger_df["tap_in_timestamp"] < SIMULATION_END_TIMESTAMP)
    ].copy()

    path_df = pd.read_csv(f"data/{data_case_name}/paths.csv")
    control.pax_path_dict, passenger_df_path = assign_passenger_path(passenger_df, path_df)

    events = pd.read_csv(f"data/{data_case_name}/events.csv")
    events = events.loc[
        (events["event_timestamp"] > SIMULATION_START_TIMESTAMP)
        & (events["event_timestamp"] < SIMULATION_END_TIMESTAMP)
    ].copy()
    control_events = select_platform_train_id(data_case_name, events)
    event_list = generate_event_list(events)

    all_platform_and_train = (
        control_events[["train_id", "platform_id"]]
        .drop_duplicates()
        .sort_values(["train_id", "platform_id"])
        .reset_index(drop=True)
    )
    vector_to_platform_map = dict(
        enumerate(
            zip(
                all_platform_and_train["train_id"],
                all_platform_and_train["platform_id"],
            )
        )
    )
    return event_list, passenger_df_path, vector_to_platform_map


def evaluate_control_vector(
    control_vector,
    iteration,
    output_case_name,
    event_list,
    passenger_df_path,
    vector_to_platform_map,
):
    control_vector = np.clip(np.asarray(control_vector, dtype=float), 0.0, 1.0)
    control_factor_dict = {
        vector_to_platform_map[i]: float(control_vector[i])
        for i in range(len(control_vector))
    }
    all_trains = {}
    all_platforms = {}
    passenger_objects = {}
    all_logs = {
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
    }
    grouped_passengers = process_passenger_group_by_origin(passenger_df_path)

    (
        _,
        _,
        passenger_objects,
        all_platforms,
        all_trains,
        all_logs,
        _,
    ) = control.simulation_with_control(
        event_list,
        passenger_objects,
        all_platforms,
        all_trains,
        all_logs,
        iteration,
        {},
        control_factor_dict,
        grouped_passengers,
        BOARD_NUM_CONTROL=False,
    )
    control.save_all_logs_with_iteration(all_logs, iteration, output_case_name)
    all_lbs = [
        all_logs["left_behind_log_temp"][pax_id]["left_behind_times"]
        for pax_id in all_logs["left_behind_log_temp"]
    ]
    return max(all_lbs)


def run_de_control(data_case_name, output_case_name, max_iter=MAX_ITER):
    event_list, passenger_df_path, vector_to_platform_map = load_simulation_context(
        data_case_name
    )
    dim = len(vector_to_platform_map)
    bounds = [(0.0, 1.0) for _ in range(dim)]
    x0 = np.ones(dim)
    iteration = {"value": 0}

    def objective(x):
        current_iteration = iteration["value"]
        if current_iteration >= max_iter:
            raise EvaluationLimitReached
        print(f"===== {output_case_name}: iteration {current_iteration} =====")
        print(f"Avg control factor: {np.mean(x):.4f}")
        value = evaluate_control_vector(
            x,
            current_iteration,
            output_case_name,
            event_list,
            passenger_df_path,
            vector_to_platform_map,
        )
        print(f"Max left-behind time: {value}")
        iteration["value"] += 1
        return value

    start_time = time.time()
    try:
        differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=10,
            workers=1,
            x0=x0,
            seed=42,
            polish=False,
        )
    except EvaluationLimitReached:
        pass
    return time.time() - start_time


def run_byo_control(data_case_name, output_case_name, max_iter=MAX_ITER):
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    event_list, passenger_df_path, vector_to_platform_map = load_simulation_context(
        data_case_name
    )
    dim = len(vector_to_platform_map)
    space = [Real(0.0, 1.0, name=f"x{i}") for i in range(dim)]
    iteration = {"value": 0}

    @use_named_args(space)
    def objective(**params):
        current_iteration = iteration["value"]
        print(f"===== {output_case_name}: iteration {current_iteration} =====")
        x = np.array([params[f"x{i}"] for i in range(dim)])
        print(f"Avg control factor: {np.mean(x):.4f}")
        value = evaluate_control_vector(
            x,
            current_iteration,
            output_case_name,
            event_list,
            passenger_df_path,
            vector_to_platform_map,
        )
        print(f"Max left-behind time: {value}")
        iteration["value"] += 1
        return value

    start_time = time.time()
    gp_minimize(objective, space, n_calls=max_iter, random_state=42)
    return time.time() - start_time


def summarize_existing_benchmarks():
    base_summary = pd.read_csv("output/network_size_experiment_summary.csv")
    rows = []
    for _, row in base_summary.iterrows():
        case_name = row["case_name"]
        mlr_best, mlr_iter = summarize_best_max_lb(f"{case_name}_MLR")
        de_best, de_iter = summarize_best_max_lb(f"{case_name}_DE")
        byo_best, byo_iter = summarize_best_max_lb(f"{case_name}_BYO")
        rows.append(
            {
                "case_name": case_name,
                "lines": row["lines"],
                "num_stations": int(row["num_stations"]),
                "num_platforms": int(row["num_platforms"]),
                "initial_max_lb": int(row["initial_max_lb"]),
                "proposed_best_max_lb": int(row["final_best_max_lb"]),
                "proposed_best_iteration": int(row["best_iteration"]),
                "mlr_best_max_lb": mlr_best,
                "mlr_best_iteration": mlr_iter,
                "de_best_max_lb": de_best,
                "de_best_iteration": de_iter,
                "byo_best_max_lb": byo_best,
                "byo_best_iteration": byo_iter,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv("output/network_size_existing_benchmark_summary.csv", index=False)
    print(summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mlr", "de", "byo", "summary"], required=True)
    parser.add_argument("--case", choices=list(NETWORK_CASES.keys()) + ["all"], default="all")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.method == "summary":
        summarize_existing_benchmarks()
        return

    cases = list(NETWORK_CASES.keys()) if args.case == "all" else [args.case]
    for case_name in cases:
        output_case_name = f"{case_name}_{args.method.upper()}"
        if args.method == "mlr":
            runtime = run_mlr_control(case_name, output_case_name, args.max_iter)
        elif args.method == "de":
            runtime = run_de_control(case_name, output_case_name, args.max_iter)
        else:
            runtime = run_byo_control(case_name, output_case_name, args.max_iter)
        best, best_iter = summarize_best_max_lb(output_case_name)
        print("SUMMARY", output_case_name, best, best_iter, runtime)


if __name__ == "__main__":
    main()
