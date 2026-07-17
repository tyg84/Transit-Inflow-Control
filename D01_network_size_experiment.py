import time
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

import _constant
import A01_generate_basic_data as basic_data
import A03_generate_events as event_data
from A04_generate_demand_data import time_multiplier, generate_individual_tap_in_time
import B03_control_strategies as control
from B01_simulation import (
    assign_passenger_path,
    generate_event_list,
    process_passenger_group_by_origin,
)


SIMULATION_START_TIMESTAMP = 6 * 3600
SIMULATION_END_TIMESTAMP = 9 * 3600
RANDOM_SEED_DEMAND = 141
RANDOM_SEED_TAP_IN = 142
MAX_ITER = 100


NETWORK_CASES = {
    "network_size_2_lines": [1, 2],
    "network_size_3_lines": [1, 2, 3],
    "network_size_4_lines": [1, 2, 3, 4],
}

LINE_LABELS = {1: "1", 2: "2", 3: "3", 4: "9"}


def _add_edge(graph, a, b, weight):
    if a == b:
        return
    graph.setdefault(a, {})
    graph.setdefault(b, {})
    if b not in graph[a] or weight < graph[a][b]:
        graph[a][b] = weight
        graph[b][a] = weight


def _single_source_dijkstra(graph, source):
    distances = {source: 0.0}
    paths = {source: [source]}
    heap = [(0.0, source)]

    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, weight in graph.get(node, {}).items():
            new_distance = distance + weight
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                paths[neighbor] = paths[node] + [neighbor]
                heapq.heappush(heap, (new_distance, neighbor))

    return paths


def _build_platform_direction_map(platform_dir_df):
    mapping = {}
    for _, row in platform_dir_df.iterrows():
        from_platform = str(row["from_platform_id"]).strip()
        to_platform = str(row["to_platform_id"]).strip()
        if "_" not in from_platform or "_" not in to_platform:
            continue
        from_station_line = "_".join(from_platform.split("_")[:2])
        to_station_line = "_".join(to_platform.split("_")[:2])
        from_direction = int(from_platform.split("_")[2])
        mapping[(from_station_line, to_station_line)] = from_direction
    return mapping


def _segment_direction_code(from_station_line, to_station_line, platform_dir_map):
    from_station = from_station_line.split("_", 1)[0]
    to_station = to_station_line.split("_", 1)[0]

    if from_station == to_station and from_station_line != to_station_line:
        return 2
    if (from_station_line, to_station_line) in platform_dir_map:
        return int(platform_dir_map[(from_station_line, to_station_line)])
    if (to_station_line, from_station_line) in platform_dir_map:
        return 1 - int(platform_dir_map[(to_station_line, from_station_line)])
    return 0


def generate_all_path_segments(case_name):
    transfer_df = pd.read_csv(f"data/{case_name}/station_line_transfer_times.csv")
    travel_df = pd.read_csv(f"data/{case_name}/station_line_travel_times.csv")
    platform_dir_df = pd.read_csv(f"data/{case_name}/platform_travel_times.csv")

    graph = {}
    for _, row in travel_df.iterrows():
        _add_edge(
            graph,
            str(row["from_station_line_id"]),
            str(row["to_station_line_id"]),
            float(row["travel_time"]),
        )
    for _, row in transfer_df.iterrows():
        _add_edge(
            graph,
            str(row["from_station_line_id"]),
            str(row["to_station_line_id"]),
            float(row["travel_time"]),
        )

    platform_dir_map = _build_platform_direction_map(platform_dir_df)
    out_rows = []
    for source in graph:
        paths = _single_source_dijkstra(graph, source)
        for destination, path in paths.items():
            if destination == source:
                continue

            cumulative_time = 0.0
            for j in range(len(path) - 1):
                from_station = path[j]
                to_station = path[j + 1]
                line_id_from = int(from_station.split("_")[1])
                segment_time = graph[from_station][to_station]
                cumulative_time += segment_time
                direction_code = _segment_direction_code(
                    from_station, to_station, platform_dir_map
                )

                if direction_code == 2:
                    if j > 0 and j < len(path) - 2:
                        from_direction_id = _segment_direction_code(
                            path[j - 1], path[j], platform_dir_map
                        )
                        to_direction_id = _segment_direction_code(
                            path[j + 1], path[j + 2], platform_dir_map
                        )
                    elif j == 0 and j < len(path) - 2:
                        to_direction_id = _segment_direction_code(
                            path[j + 1], path[j + 2], platform_dir_map
                        )
                        from_direction_id = to_direction_id
                    elif j == len(path) - 2:
                        from_direction_id = _segment_direction_code(
                            path[j - 1], path[j], platform_dir_map
                        )
                        to_direction_id = from_direction_id
                    else:
                        from_direction_id = 0
                        to_direction_id = 0
                else:
                    from_direction_id = int(direction_code)
                    to_direction_id = int(direction_code)

                out_rows.append(
                    {
                        "origin": source,
                        "destination": destination,
                        "path_id": 1,
                        "line_id": line_id_from,
                        "from_direction_id": from_direction_id,
                        "to_direction_id": to_direction_id,
                        "if_transfer": 1 if direction_code == 2 else 0,
                        "from_station": from_station,
                        "to_station": to_station,
                        "cumulated_travel_time": cumulative_time,
                    }
                )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "origin",
            "destination",
            "path_id",
            "line_id",
            "from_direction_id",
            "to_direction_id",
            "if_transfer",
            "from_station",
            "to_station",
            "cumulated_travel_time",
        ],
    )
    out_df.to_csv(f"data/{case_name}/paths.csv", index=False)


def generate_demand_data_from_stations(stations, case_name, demand_factor, random_seed):
    time_bin = 900
    day_end = 24 * 3600
    alpha = _constant.DEMAND_RATE * demand_factor
    records = []

    stations = stations.rename(columns={"station id": "station_id"}).copy()

    for t in range(0, day_end, time_bin):
        np.random.seed(random_seed + t)
        sudden_peak_prob = 0.05
        mult = time_multiplier(t) * np.random.choice(
            [1, 2], p=[1 - sudden_peak_prob, sudden_peak_prob]
        )

        for o in stations.itertuples():
            for d in stations.itertuples():
                if o.station_id == d.station_id or o.Line != d.Line:
                    continue
                base = o.poprating * d.workrating
                num = int(alpha * base * mult)
                if num > 0:
                    records.append(
                        [
                            o.station_id,
                            d.station_id,
                            t,
                            t + time_bin,
                            num,
                        ]
                    )

    od_df = pd.DataFrame(
        records,
        columns=[
            "origin_station_id",
            "destination_station_id",
            "tap_in_time_start",
            "tap_in_time_end",
            "num_passengers",
        ],
    )
    od_df.to_csv(f"data/{case_name}/demands.csv", index=False)


def prepare_case(case_name, selected_lines, demand_factor=1.0):
    Path(f"data/{case_name}").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv("data/manual_input_data/testSubwayStation.csv")
    raw = raw.loc[raw["Line"].isin(selected_lines)].copy()

    basic_data.generate_platforms(raw, case_name)
    basic_data.generate_station_pair_travel_time(raw, case_name)
    platforms = pd.read_csv(f"data/{case_name}/platforms.csv")
    basic_data.construct_transfer_time(platforms, case_name)
    generate_all_path_segments(case_name)

    headway = pd.read_csv("data/manual_input_data/headway.csv")
    headway = headway.loc[headway["line_id"].isin(selected_lines)].copy()
    platform_travel_times = pd.read_csv(f"data/{case_name}/platform_travel_times.csv")
    event_data.case_name = case_name
    event_data.generate_events(headway, platforms, platform_travel_times)

    train_capacity_df = pd.read_csv("data/manual_input_data/train_capacity.csv")
    train_capacity_df = train_capacity_df.loc[
        train_capacity_df["line_id"].isin(selected_lines)
    ].copy()
    train_capacity_df["train_capacity"] = np.round(
        train_capacity_df["train_capacity"] * _constant.TRAIN_CAPACITY_FACTOR
    ).astype(int)
    train_capacity_df.to_csv(f"data/{case_name}/train_capacity_adjusted.csv", index=False)

    generate_demand_data_from_stations(
        raw, case_name, demand_factor=demand_factor, random_seed=RANDOM_SEED_DEMAND
    )
    demands = pd.read_csv(f"data/{case_name}/demands.csv")
    generate_individual_tap_in_time(
        demands, case_name, random_seed=RANDOM_SEED_TAP_IN
    )


def run_proposed_control(case_name, max_iter=MAX_ITER):
    control.SIMULATION_START_TIMESTAMP = SIMULATION_START_TIMESTAMP

    train_capacity_df = pd.read_csv(f"data/{case_name}/train_capacity_adjusted.csv")
    control.train_capacity_dict = train_capacity_df.set_index(
        ["line_id", "direction_id"]
    )["train_capacity"].to_dict()

    passenger_df = pd.read_csv(f"data/{case_name}/individual_demands.csv")
    passenger_df = passenger_df.loc[
        (passenger_df["tap_in_timestamp"] > SIMULATION_START_TIMESTAMP)
        & (passenger_df["tap_in_timestamp"] < SIMULATION_END_TIMESTAMP)
    ].copy()

    path_df = pd.read_csv(f"data/{case_name}/paths.csv")
    control.pax_path_dict, passenger_df_path = assign_passenger_path(passenger_df, path_df)

    events = pd.read_csv(f"data/{case_name}/events.csv")
    events = events.loc[
        (events["event_timestamp"] > SIMULATION_START_TIMESTAMP)
        & (events["event_timestamp"] < SIMULATION_END_TIMESTAMP)
    ].copy()
    event_list = generate_event_list(events)

    control_board_num_dict = {}
    best_results = {"best_iteration": 0, "best_max_LB": np.inf}

    start_time = time.time()
    for iteration in range(max_iter):
        print(f"===== {case_name}: iteration {iteration} =====")
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
            control_board_num_dict,
            _,
            passenger_objects,
            all_platforms,
            all_trains,
            all_logs,
            platform_line_index,
        ) = control.simulation_with_control(
            event_list,
            passenger_objects,
            all_platforms,
            all_trains,
            all_logs,
            iteration,
            control_board_num_dict,
            None,
            grouped_passengers,
            BOARD_NUM_CONTROL=True,
        )
        control_board_num_dict, stop, best_results = control.update_control_strategy(
            all_logs,
            all_trains,
            all_platforms,
            passenger_objects,
            control_board_num_dict,
            platform_line_index,
            best_results,
            iteration,
        )
        control.save_all_logs_with_iteration(all_logs, iteration, case_name)
        print(f"Best results: {best_results}")
        if stop:
            break

    return time.time() - start_time


def update_mlr_control_strategy(all_logs, all_trains, all_platforms, control_board_num_dict):
    lb_log_df = pd.DataFrame(all_logs["left_behind_log"])
    max_lb_pax_times = lb_log_df["left_behind_times"].max()
    if max_lb_pax_times <= 1:
        return control_board_num_dict, True

    max_lb_pax = lb_log_df.loc[
        lb_log_df["left_behind_times"] == max_lb_pax_times
    ].copy()
    max_lb_pax = max_lb_pax.sort_values(
        ["boarded_train_id", "boarded_platform_seq"], ascending=[True, False]
    )

    marginal_reduction = 0.05
    for boarded_train_id, pax_group in max_lb_pax.groupby("boarded_train_id"):
        previous_train_id = pax_group["previous_train_id"].iloc[0]
        platform_id = pax_group["boarding_platform"].iloc[0]
        if pd.isna(previous_train_id) or previous_train_id not in all_trains:
            continue

        reserved_space = int(np.max(pax_group["seq_at_queue_when_board"])) + 1
        previous_train = all_trains[previous_train_id]
        passed_platforms = list(previous_train.remaining_passenger_at_each_platform.keys())
        upstream_platforms = [
            (plat_id, all_platforms[plat_id].platform_seq)
            for plat_id in passed_platforms
            if all_platforms[plat_id].platform_seq < all_platforms[platform_id].platform_seq
        ]
        upstream_platforms = sorted(upstream_platforms, key=lambda x: -x[1])

        cumulative_remove = 0
        for upstream_platform_id, _ in upstream_platforms:
            old_num_board = control_board_num_dict[
                (previous_train_id, upstream_platform_id)
            ]["Onboard"]
            if old_num_board > 0:
                reduction = max(1, int(old_num_board * marginal_reduction))
                new_num_board = old_num_board - reduction
            else:
                reduction = 0
                new_num_board = old_num_board
            control_board_num_dict[(previous_train_id, upstream_platform_id)][
                "Control"
            ] = new_num_board
            cumulative_remove += reduction
            if cumulative_remove >= reserved_space:
                break

        if control_board_num_dict[(previous_train_id, platform_id)]["Control"] is not None:
            control_board_num_dict[(previous_train_id, platform_id)][
                "Control"
            ] += cumulative_remove

    return control_board_num_dict, False


def run_mlr_control(data_case_name, output_case_name, max_iter=MAX_ITER):
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
    event_list = generate_event_list(events)

    control_board_num_dict = {}
    start_time = time.time()
    for iteration in range(max_iter):
        output_file = (
            Path("output")
            / output_case_name
            / f"left_behind_log_iteration_{iteration}.csv"
        )
        print(f"===== {output_case_name}: iteration {iteration} =====")
        if output_file.exists():
            print(f"Skip existing iteration {iteration}")
            continue
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
            control_board_num_dict,
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
            control_board_num_dict,
            None,
            grouped_passengers,
            BOARD_NUM_CONTROL=True,
        )
        control.save_all_logs_with_iteration(all_logs, iteration, output_case_name)
        control_board_num_dict, stop = update_mlr_control_strategy(
            all_logs, all_trains, all_platforms, control_board_num_dict
        )
        if stop:
            break

    return time.time() - start_time


def summarize_case(case_name, selected_lines, runtime_sec=None):
    data_dir = Path("data") / case_name
    output_dir = Path("output") / case_name

    platforms = pd.read_csv(data_dir / "platforms.csv")
    stations = platforms["station_id"].nunique()
    platform_count = len(platforms)
    passengers = pd.read_csv(data_dir / "individual_demands.csv")
    passengers = passengers.loc[
        (passengers["tap_in_timestamp"] > SIMULATION_START_TIMESTAMP)
        & (passengers["tap_in_timestamp"] < SIMULATION_END_TIMESTAMP)
    ]
    events = pd.read_csv(data_dir / "events.csv")
    events = events.loc[
        (events["event_timestamp"] > SIMULATION_START_TIMESTAMP)
        & (events["event_timestamp"] < SIMULATION_END_TIMESTAMP)
    ]

    iterations = []
    for file_path in output_dir.glob("left_behind_log_iteration_*.csv"):
        iterations.append(int(file_path.stem.rsplit("_", 1)[1]))
    iterations = sorted(iterations)

    best_max = np.inf
    best_iter = None
    final_iter = iterations[-1]
    initial_df = pd.read_csv(output_dir / "left_behind_log_iteration_0.csv")
    final_df = pd.read_csv(output_dir / f"left_behind_log_iteration_{final_iter}.csv")

    for iteration in iterations:
        lb = pd.read_csv(output_dir / f"left_behind_log_iteration_{iteration}.csv")
        max_lb = int(lb["left_behind_times"].max())
        if max_lb < best_max:
            best_max = max_lb
            best_iter = iteration

    return {
        "case_name": case_name,
        "lines": "+".join(LINE_LABELS.get(line, str(line)) for line in selected_lines),
        "num_lines": len(selected_lines),
        "num_stations": int(stations),
        "num_platforms": int(platform_count),
        "num_passengers": int(len(passengers)),
        "num_events": int(len(events)),
        "initial_max_lb": int(initial_df["left_behind_times"].max()),
        "final_best_max_lb": int(best_max),
        "best_iteration": int(best_iter),
        "final_iteration": int(final_iter),
        "initial_mean_lb": float(initial_df["left_behind_times"].mean()),
        "final_mean_lb": float(final_df["left_behind_times"].mean()),
        "initial_positive_records": int((initial_df["left_behind_times"] > 0).sum()),
        "final_positive_records": int((final_df["left_behind_times"] > 0).sum()),
        "runtime_sec": runtime_sec,
    }


def summarize_best_max_lb(output_case_name):
    output_dir = Path("output") / output_case_name
    iterations = []
    for file_path in output_dir.glob("left_behind_log_iteration_*.csv"):
        iterations.append(int(file_path.stem.rsplit("_", 1)[1]))
    iterations = sorted(iterations)
    if not iterations:
        return np.nan, np.nan

    best_max = np.inf
    best_iter = None
    for iteration in iterations:
        lb = pd.read_csv(output_dir / f"left_behind_log_iteration_{iteration}.csv")
        max_lb = int(lb["left_behind_times"].max())
        if max_lb < best_max:
            best_max = max_lb
            best_iter = iteration
    return int(best_max), int(best_iter)


def main():
    rows = []
    for case_name, selected_lines in NETWORK_CASES.items():
        prepare_case(case_name, selected_lines)
        runtime_sec = run_proposed_control(case_name)
        rows.append(summarize_case(case_name, selected_lines, runtime_sec=runtime_sec))

    summary = pd.DataFrame(rows)
    Path("output").mkdir(exist_ok=True)
    summary.to_csv("output/network_size_experiment_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
