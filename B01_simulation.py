import pandas as pd
import numpy as np
import _constant
import copy
import os
import time

class Event:
    def __init__(self, line_id, direction_id, station_id, platform_id, train_id, event_timestamp, event_type):
        self.line_id = line_id
        self.direction_id = direction_id
        self.station_id = station_id
        self.platform_id = platform_id
        self.train_id = train_id
        self.event_timestamp = event_timestamp
        self.event_type = event_type


class Train:
    def __init__(self, train_id, line_id, direction_id):
        self.train_id = train_id
        self.capacity = train_capacity_dict[(line_id, direction_id)]
        self.passenger_list = []


class Platform:
    def __init__(self, platform_id, line_id, direction_id):
        self.platform_id = platform_id
        self.line_id = line_id
        self.direction_id = direction_id
        self.platform_station_line_id = self.platform_id.split('_')[0] + '_' + self.platform_id.split('_')[1]
        self.passenger_list = []
        self.last_train_departure_time = SIMULATION_START_TIMESTAMP

    def add_passenger(self, passenger):
        self.passenger_list.append(passenger)


class Passenger:
    def __init__(self, origin_station_id, destination_station_id, passenger_id):
        self.origin_station_id = origin_station_id
        self.destination_station_id = destination_station_id
        self.passenger_id = passenger_id
        self.origin = pax_path_dict[passenger_id]['origin']
        self.destination = pax_path_dict[passenger_id]['destination']
        self.path_id = pax_path_dict[passenger_id]['path_id']
        self.transfer_from_platform_list = pax_path_dict[passenger_id]['from_transfer_platforms'].split(',')
        self.transfer_to_platform_list = pax_path_dict[passenger_id]['to_transfer_platforms'].split(',')
        self.origin_line_id = int(pax_path_dict[passenger_id]['origin'].split('_')[1])
        self.trajectory = {'trajectory_type': [], 'trajectory_time': []}
        self.tap_in_timestamp = None
        self.left_behind_times = {}


def generate_event_list(events):
    event_list = []
    events = events.sort_values(['event_timestamp'])
    for line_id, direction_id, station_id, platform_id, train_id, event_timestamp, event_type in zip(
            events['line_id'], events['direction_id'], events['station_id'],
            events['platform_id'], events['train_id'],
            events['event_timestamp'], events['event_type']):
        event_list.append(Event(line_id, direction_id, station_id, platform_id, train_id, event_timestamp, event_type))
    return event_list


def offload_passengers(event):
    train = all_trains[event.train_id]
    if len(train.passenger_list) == 0:
        return
    platform = all_platforms[event.platform_id]
    remained_passengers = []
    for p in train.passenger_list:
        if len(p.transfer_from_platform_list) and platform.platform_id == p.transfer_from_platform_list[0]:
            # transfer
            p.trajectory['trajectory_type'].append('Transfer')
            p.trajectory['trajectory_time'].append(event.event_timestamp + _constant.DEFAULT_EXIT_WALKING_TIME)
            # next transfer platform
            next_platform_id = p.transfer_from_platform_list.pop(0)
            all_platforms[next_platform_id].passenger_list.append(p)
        elif p.destination_station_id == platform.platform_station_line_id:
            # leaving passenger
            p.trajectory['trajectory_type'].append('Exit')
            p.trajectory['trajectory_time'].append(event.event_timestamp + _constant.DEFAULT_EXIT_WALKING_TIME)
            # log passenger travel time
            trajectory_log['passenger_id'] += [p.passenger_id] * len(p.trajectory['trajectory_type'])
            trajectory_log['trajectory_type'] += p.trajectory['trajectory_type']
            trajectory_log['trajectory_time'] += p.trajectory['trajectory_time']
        else:
            remained_passengers.append(p)

    train.passenger_list = copy.deepcopy(remained_passengers)


def onboard_passengers(event):
    train = all_trains[event.train_id]
    platform = all_platforms[event.platform_id]
    available_space = train.capacity - len(train.passenger_list)
    num_onboard_pax = min(available_space, len(platform.passenger_list))
    to_board_passengers = platform.passenger_list[:num_onboard_pax]
    left_behind_passengers = platform.passenger_list[num_onboard_pax:]

    for p in to_board_passengers:
        p.trajectory['trajectory_type'].append('Boarding')
        p.trajectory['trajectory_time'].append(event.event_timestamp)
        # passenger_travel_log.append({
        #     'passenger_id': p.passenger_id,
        #     'origin_station_id': p.origin_station_id,
        #     'destination_station_id': p.destination_station_id,
        #     'boarding_time': event.event_timestamp,
        #     'alighting_time': None,
        #     'total_travel_time': None
        # })

    for p in left_behind_passengers:
        pass

    train.passenger_list.extend(to_board_passengers)
    platform.passenger_list = platform.passenger_list[available_space:]

    # Log train load and platform queue
    # train_load_log.append({
    #     'train_id': train.train_id,
    #     'timestamp': event.event_timestamp,
    #     'train_load': len(train.passenger_list)
    # })
    # platform_queue_log.append({
    #     'platform_id': platform.platform_id,
    #     'timestamp': event.event_timestamp,
    #     'queue_length': len(platform.passenger_list)
    # })


def add_new_passengers_to_platform(event):
    platform_id = event.platform_id
    platform = all_platforms[platform_id]

    start_time = platform.last_train_departure_time
    end_time = event.event_timestamp
    station_id = event.station_id

    pax_group = grouped_passengers.get(station_id, None)
    if pax_group is None:
        return

    if isinstance(pax_group, dict) and 'passenger_id' in pax_group:
        pid_list = pax_group['passenger_id']
        tap_list = pax_group['tap_in_timestamp']
        dest_list = pax_group['destination_station_id']

        tap_arr = np.array(tap_list)
        i = int(tap_arr.searchsorted(start_time, side="left"))
        j = int(tap_arr.searchsorted(end_time, side="left"))
        if i >= j:
            return

        for k in range(i, j):
            pid = int(pid_list[k])
            dest = int(dest_list[k])
            tap_ts = int(tap_list[k])
            p = Passenger(station_id, dest, pid)
            p.tap_in_timestamp = tap_ts
            passenger_objects[pid] = p
            platform.add_passenger(p)

        pax_group['passenger_id'] = pid_list[j:]
        pax_group['tap_in_timestamp'] = tap_list[j:]
        pax_group['destination_station_id'] = dest_list[j:]
        grouped_passengers[station_id] = pax_group
        return

    if isinstance(pax_group, list):
        remaining = []
        for entry in pax_group:
            if isinstance(entry, dict):
                try:
                    pid = int(entry['passenger_id'])
                    tap_ts = int(entry['tap_in_timestamp'])
                    dest = int(entry['destination_station_id'])
                except Exception:
                    remaining.append(entry)
                    continue
                if start_time <= tap_ts < end_time:
                    p = Passenger(station_id, dest, pid)
                    p.tap_in_timestamp = tap_ts
                    passenger_objects[pid] = p
                    platform.add_passenger(p)
                else:
                    remaining.append(entry)
            elif hasattr(entry, 'tap_in_timestamp'):
                if start_time <= entry.tap_in_timestamp < end_time:
                    pid = int(entry.passenger_id)
                    passenger_objects[pid] = entry
                    platform.add_passenger(entry)
                else:
                    remaining.append(entry)
            else:
                remaining.append(entry)
        grouped_passengers[station_id] = remaining
        return


def initialize_trains(event):
    if event.train_id not in all_trains:
        all_trains[event.train_id] = Train(event.train_id, event.line_id, event.direction_id)


def initialize_platforms(event):
    if event.platform_id not in all_platforms:
        all_platforms[event.platform_id] = Platform(event.platform_id, event.line_id, event.direction_id)


def simulation_main(event_list):

    s_time = time.time()
    for event_id, event in enumerate(event_list):
        if event_id > 0 and event_id % 1000 == 0:
            print('Start simulation event #{}, total {}'.format(event_id, len(event_list)))
            total_spent_time = time.time() - s_time
            estimate_total_finish_time = len(event_list) * (total_spent_time / event_id)
            print(f'estimate_total_finish_time: {round(estimate_total_finish_time)} sec')
        initialize_trains(event)
        initialize_platforms(event)

        if event.event_type == 'Arrival':
            offload_passengers(event)
        elif event.event_type == 'Departure':
            add_new_passengers_to_platform(event)
            onboard_passengers(event)

    print('total spent time: {} sec'.format(round(time.time() - s_time)))

    # Save output metrics
    # os.makedirs('data', exist_ok=True)
    # pd.DataFrame(train_load_log).to_csv('data/train_load.csv', index=False)
    # pd.DataFrame(platform_queue_log).to_csv('data/platform_queue.csv', index=False)
    # pd.DataFrame(left_behind_log).to_csv('data/left_behind.csv', index=False)
    # pd.DataFrame(passenger_travel_log).to_csv('data/passenger_travel_times.csv', index=False)
    print("Success")


def process_passenger_group_by_origin(passenger_df):
    passenger_df = passenger_df.sort_values("tap_in_timestamp")
    all_passengers = (
        passenger_df.groupby('origin_station_id')[['passenger_id', 'destination_station_id', 'tap_in_timestamp']]
        .apply(lambda x: x.to_dict(orient='list'))
        .to_dict()
    )
    return all_passengers


def assign_passenger_path(passenger_df, path_df):
    path_df['origin_station_id'] = path_df['origin'].str.split('_').str[0].astype('int')
    path_df['destination_station_id'] = path_df['destination'].str.split('_').str[0].astype('int')
    path_df['from_platform_id'] = path_df['from_station'] + '_' + path_df['from_direction_id'].astype(str)
    path_df['to_platform_id'] = path_df['to_station'] + '_' + path_df['to_direction_id'].astype(str)
    unique_path = path_df.groupby(['origin_station_id', 'destination_station_id', 'origin', 'destination', 'path_id']).agg(
        total_travel_time=('cumulated_travel_time', 'last'),
        from_transfer_platforms=('from_platform_id', lambda x: ",".join(x[path_df.loc[x.index, 'if_transfer'].eq(1)].unique())),
        to_transfer_platforms=('to_platform_id', lambda x: ",".join(x[path_df.loc[x.index, 'if_transfer'].eq(1)].unique()))
    ).reset_index()

    unique_path = unique_path.sort_values(['total_travel_time'], ascending=True)
    unique_path = unique_path.groupby(['origin_station_id', 'destination_station_id', 'path_id']).first().reset_index()

    passenger_df_path = passenger_df.merge(unique_path, on=['origin_station_id', 'destination_station_id'])
    check_path_num = passenger_df_path.groupby(['passenger_id'])['path_id'].count().reset_index()
    check_path_num = check_path_num.loc[check_path_num['path_id'] > 1]
    assert len(check_path_num) == 0

    pax_path_dict = passenger_df_path[['passenger_id','origin','destination','path_id','from_transfer_platforms','to_transfer_platforms']].set_index('passenger_id').to_dict(orient='index')

    return pax_path_dict


#######################
# MAIN
#######################

if __name__ == '__main__':
    # train_load_log = []  # Each departure: train_id, timestamp, load
    # platform_queue_log = []  # At each event: platform_id, timestamp, queue_length
    # left_behind_log = []  # Track passengers who could not board train
    # passenger_travel_log = []  # For each passenger: id, origin, destination, boarding_time, alighting_time, total_travel_time
    trajectory_log = {'passenger_id': [], 'trajectory_type': [], 'trajectory_time': []}

    SIMULATION_START_TIMESTAMP = 5 * 3600
    SIMULATION_END_TIMESTAMP = 24 * 3600

    train_capacity_df = pd.read_csv('data/train_capacity.csv')
    train_capacity_dict = train_capacity_df.set_index(['line_id', 'direction_id'])['train_capacity'].to_dict()

    passenger_df = pd.read_csv('data/individual_demands.csv')
    passenger_df = passenger_df.loc[
        (passenger_df['tap_in_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (passenger_df['tap_in_timestamp'] < SIMULATION_END_TIMESTAMP)
    ]

    grouped_passengers = process_passenger_group_by_origin(passenger_df)
    passenger_objects = {}

    path_df = pd.read_csv('data/paths.csv')
    pax_path_dict = assign_passenger_path(passenger_df, path_df)
    print('Finish assigning passenger paths...')

    all_trains = {}
    all_platforms = {}

    events = pd.read_csv('data/events.csv')
    events = events.loc[
        (events['event_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (events['event_timestamp'] < SIMULATION_END_TIMESTAMP)
    ]

    event_list = generate_event_list(events)
    print('Finish generating event list...')
    simulation_main(event_list)
