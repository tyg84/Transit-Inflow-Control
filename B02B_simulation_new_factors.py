"""
B02_experiments.py
------------------
Admission-control experiment runner WITHOUT importing B01 or any other project file.
You inject the required simulator functions & data when calling run_experiments().

This module provides:

 - StationOutside
 - AdmissionPolicy
 - add_new_passengers_controlled
 - run_experiments()

Nothing in this file imports your simulator.
Nothing in this file modifies your simulator files.

Your simulator only needs to supply:
    - simulation_main()
    - generate_event_list()
    - assign_passenger_path()
    - process_passenger_group_by_origin()
    - train_capacity_dict (dict)
    - all_trains (dict)
    - all_platforms (dict)
    - passenger_objects (dict)
    - trajectory_log (dict)
and it must call `add_new_passengers_to_platform()` at the correct moments.

You will patch that call with our controlled version.
"""

import math
import random
from collections import defaultdict, deque
import pandas as pd
import numpy as np


# --------------------------------------------------------
#  StationOutside
# --------------------------------------------------------
class StationOutside:
    def __init__(self, grouped):
        self.arrivals = grouped
        self.waiting = defaultdict(deque)
        self.next_idx = {sid: 0 for sid in grouped}
        self.abandoned = []
        self.total_admitted = 0

    def update_until(self, t):
        for sid, arrivals in self.arrivals.items():
            idx = self.next_idx[sid]
            while idx < len(arrivals) and arrivals[idx]["tap_in_timestamp"] <= t:
                p = arrivals[idx].copy()
                p["first_seen"] = p["tap_in_timestamp"]
                self.waiting[sid].append(p)
                idx += 1
            self.next_idx[sid] = idx

    def admit(self, sid, k, t):
        q = self.waiting[sid]
        admitted = []
        for _ in range(min(k, len(q))):
            p = q.popleft()
            p["admit_time"] = t
            admitted.append(p)
            self.total_admitted += 1
        return admitted

    def remaining_waits(self, T):
        out = []
        for sid, q in self.waiting.items():
            for p in q:
                out.append(T - p["first_seen"])
        return out


# --------------------------------------------------------
#  AdmissionPolicy
# --------------------------------------------------------
class AdmissionPolicy:
    def __init__(self, mode):
        self.mode = mode

    def max_admit(self, train_cap, onboard, cd_flag=False):
        if self.mode == "no_control":
            return None

        if self.mode == "fixed":
            return max(1, int(0.20 * train_cap))

        if self.mode == "dynamic":
            return int(0.50 * max(0, train_cap - onboard))

        if self.mode == "collector_distributor":
            if cd_flag:
                return None
            return max(1, int(0.20 * train_cap))

        return None


# --------------------------------------------------------
#  Controlled passenger admission wrapper
# --------------------------------------------------------
def add_new_passengers_controlled(
    event,
    station_outside,
    platforms,
    passenger_objects,
    policy,
    train_lookup,
    cd_map,
):
    t = event.event_timestamp
    sid = event.station_id
    pid = event.platform_id
    tid = event.train_id

    station_outside.update_until(t)

    train = train_lookup(tid)
    if train:
        cap = train.capacity
        onboard = len(train.passenger_list)
    else:
        cap = 9999
        onboard = 0

    cd_flag = cd_map.get(pid, False)
    k = policy.max_admit(cap, onboard, cd_flag)

    need = max(0, cap - onboard)
    if k is not None:
        need = min(need, k)

    new = station_outside.admit(sid, need, t)

    for p in new:
        obj = passenger_objects[p["passenger_id"]]
        platforms[pid].add_passenger(obj)

    return [p["passenger_id"] for p in new]


# --------------------------------------------------------
#  Main experiment runner — NO imports, fully standalone
# --------------------------------------------------------
def run_experiments(
    *,
    # injected simulator functions
    generate_event_list,
    simulation_main,
    assign_passenger_path,
    process_passenger_group_by_origin,

    # injected simulator state containers
    trajectory_log,
    all_trains,
    all_platforms,
    passenger_objects,
    train_capacity_dict,

    # input datasets
    events_df,
    platforms_df,
    travel_times_df,
    path_df,
    passenger_df,

    sim_start,
    sim_end,
):
    # Build demand group
    grouped = process_passenger_group_by_origin(passenger_df)
    station_outside = StationOutside(grouped)

    # Pre-build event list
    events = generate_event_list(events_df)

    # Precompute collector/distributor mapping
    cd_map = {}
    for _, row in platforms_df.iterrows():
        cd_map[row["platform_id"]] = False  # you replace with your logic

    # Build passenger paths
    pax_paths = assign_passenger_path(passenger_df, path_df)

    strategies = [
        "no_control",
        "fixed",
        "dynamic",
        "collector_distributor",
    ]

    results = []

    for mode in strategies:
        print(f"\n=== Running {mode} ===")

        # Reset sim global structures
        trajectory_log.clear()
        all_trains.clear()
        all_platforms.clear()
        passenger_objects.clear()

        # Create objects for each passenger
        for _, row in passenger_df.iterrows():
            passenger_objects[row.passenger_id] = row  # or your class

        policy = AdmissionPolicy(mode)

        # Patch admission call
        def admission_wrapper(event):
            return add_new_passengers_controlled(
                event,
                station_outside,
                all_platforms,
                passenger_objects,
                policy,
                train_lookup=lambda tid: all_trains.get(tid),
                cd_map=cd_map,
            )

        # Run simulation
        simulation_main(
            events,
            add_new_passengers_to_platform=admission_wrapper,
            pax_paths=pax_paths,
            sim_start=sim_start,
            sim_end=sim_end,
        )

        # Compute metrics
        exited = len([i for i, t in zip(
            trajectory_log["passenger_id"],
            trajectory_log["trajectory_type"]
        ) if t == "Exit"])

        rem_wait = station_outside.remaining_waits(sim_end)
        max_wait = max(rem_wait) if len(rem_wait) else 0

        results.append({
            "strategy": mode,
            "exited": exited,
            "max_left_behind": max_wait,
        })

    return pd.DataFrame(results)
