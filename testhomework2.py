import pandas as pd
import numpy as np
from collections import defaultdict


def generate_event_list(events):
    events = events.sort_values(['event_timestamp'])
    return events.to_dict('records')


def initialize_train_capacities(train_capacity_df):
    return train_capacity_df.set_index(['line_id', 'direction_id'])['train_capacity'].to_dict()


def initialize_platforms(events):
    platforms = {}
    for platform_id in events['platform_id'].unique():
        platforms[platform_id] = []
    return platforms


def offload_passengers(event, trains, platforms, paths_df, passenger_records):
    train_id = event['train_id']
    if train_id not in trains:
        return

    offloaded = []
    remaining = []

    for p in trains[train_id]:
        if str(event['station_id']) == str(p['destination']).split('_')[0]:
            p['exit_time'] = event['event_timestamp']
            passenger_records.append(p)
            offloaded.append(p)
        else:
            path_rows = paths_df[
                (paths_df['origin'] == p['origin']) &
                (paths_df['destination'] == p['destination']) &
                (paths_df['path_id'] == p['path_id'])
            ]
            transfer_row = path_rows[path_rows['from_station'].str.startswith(str(event['station_id']))]
            if not transfer_row.empty:
                next_row = transfer_row.iloc[0]
                next_platform = f"{next_row['to_station'].split('_')[0]}_{next_row['line_id']}_{next_row['direction_id']}"
                p['currentplatform'] = next_platform
                platforms[next_platform].append(p)
                offloaded.append(p)
            else:
                remaining.append(p)

    trains[train_id] = remaining


def onboard_passengers(event, trains, platforms, train_capacities):
    train_id = event['train_id']
    platform_id = event['platform_id']
    key = (event['line_id'], event['direction_id'])
    capacity = train_capacities.get(key, 2000)

    if train_id not in trains:
        trains[train_id] = []

    available_space = capacity - len(trains[train_id])
    if available_space <= 0:
        return

    if platform_id not in platforms or len(platforms[platform_id]) == 0:
        return

    to_board = platforms[platform_id][:available_space]
    for p in to_board:
        p['board_time'] = event['event_timestamp']

    trains[train_id].extend(to_board)
    platforms[platform_id] = platforms[platform_id][available_space:]


def simulation_main(events, paths_df, train_capacity_df, platforms=None):
    train_capacities = initialize_train_capacities(train_capacity_df)
    if platforms is None:
        platforms = initialize_platforms(events)
    trains = defaultdict(list)
    passenger_records = []

    total_events = 0
    arrivals = 0
    departures = 0

    for event in generate_event_list(events):
        total_events += 1
        if event['event_type'] == 'Arrival':
            arrivals += 1
            offload_passengers(event, trains, platforms, paths_df, passenger_records)
        elif event['event_type'] == 'Departure':
            departures += 1
            onboard_passengers(event, trains, platforms, train_capacities)

    if len(passenger_records) > 0:
        df = pd.DataFrame(passenger_records)
        df.to_csv('data/simulation_passengers.csv', index=False)
    else:
        pd.DataFrame(columns=['passenger_id', 'origin', 'destination', 'path_id', 'entry_time', 'exit_time']).to_csv(
            'data/simulation_passengers.csv', index=False
        )

    # Console output (unchanged)
    print("SUccess")
    print("Total events processed: " + str(total_events))
    print("Arrived events: " + str(arrivals))
    print("Departure events: " + str(departures))
    print("Passenger records saved: " + str(len(passenger_records)))


def generate_mock_passengers(final_platform_queues):
    passengers = []
    pid = 0
    for _, row in final_platform_queues.iterrows():
        queue_len = int(row['queue_len'])
        if queue_len == 0:
            continue
        for _ in range(queue_len):
            pid += 1
            passengers.append({
                'passenger_id': pid,
                'origin': f"{row['station_id']}_{row['line']}",
                'destination': f"{row['station_id'] + 5}_{row['line']}",
                'path_id': 1,
                'entry_time': 0,
                'exit_time': None,
                'currentplatform': f"{row['station_id']}_{row['line']}_{row['direction']}",
                'board_time': None
            })
    return passengers


if __name__ == '__main__':
    # Load data
    events = pd.read_csv('data/events.csv')
    paths = pd.read_csv('data/Paths.csv')
    train_capacity = pd.read_csv('data/train_capacity.csv')
    # final_platform_queues = pd.read_csv('data/final_platform_queues.csv')

    # # Initialize platforms and inject passengers
    # initial_platforms = initialize_platforms(events)
    # initial_passengers = generate_mock_passengers()
    #
    # for p in initial_passengers:
    #     if p['currentplatform'] in initial_platforms:
    #         initial_platforms[p['currentplatform']].append(p)

    # Run simulation with initialized passengers
    simulation_main(events, paths, train_capacity, platforms=initial_platforms)
