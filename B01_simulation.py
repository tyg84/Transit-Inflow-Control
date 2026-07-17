from collections import OrderedDict

import pandas as pd
import numpy as np
import _constant
import copy

class Event:
    def __init__(self, line_id, direction_id, station_id, platform_id, platform_seq_no, train_id, event_timestamp, event_type, event_id):
        self.line_id = line_id
        self.direction_id = direction_id
        self.station_id = station_id
        self.platform_id = platform_id
        self.train_id = train_id
        self.event_timestamp = event_timestamp
        self.event_type = event_type
        self.event_id = event_id
        self.platform_seq_no = platform_seq_no


class Train:
    def __init__(self, train_id, line_id, direction_id, train_capacity_dict):
        self.train_id = train_id
        self.capacity = train_capacity_dict[(line_id, direction_id)]
        self.passenger_list = []
        self.remaining_passenger_at_each_platform = {} ## grouped by boarding stations
        self.available_capacity_at_each_platform_departure = {}


class Platform:
    def __init__(self, platform_id, line_id, direction_id, seq_no, SIMULATION_START_TIMESTAMP):
        self.platform_id = platform_id
        self.line_id = line_id
        self.direction_id = direction_id
        self.platform_station_line_id = self.platform_id.split('_')[0] + '_' + self.platform_id.split('_')[1]
        self.platform_seq = seq_no
        self.passenger_list = []
        self.last_train_departure_time = SIMULATION_START_TIMESTAMP


    def add_passenger(self, passenger):
        self.passenger_list.append(passenger)


class Passenger:
    def __init__(self, origin_station_id, destination_station_id, passenger_id, origin_platform_id, pax_path_dict):
        self.origin_station_id = origin_station_id
        self.destination_station_id = destination_station_id
        self.passenger_id = passenger_id
        self.origin = pax_path_dict[passenger_id]['origin']
        self.destination = pax_path_dict[passenger_id]['destination']
        self.path_id = pax_path_dict[passenger_id]['path_id']
        self.transfer_from_platform_list = pax_path_dict[passenger_id]['from_transfer_platforms'].split(',')
        self.transfer_to_platform_list = pax_path_dict[passenger_id]['to_transfer_platforms'].split(',')
        self.origin_line_id = int(pax_path_dict[passenger_id]['origin'].split('_')[1])
        self.trajectory = {
            'trajectory_type': [],
            'trajectory_time': [],
            'trajectory_platform': [],
            'trajectory_event_id': [],
        }
        self.boarding_seq_records = {} ## platform: seq in the queue
        self.tap_in_timestamp = None
        self.left_behind_times = {origin_platform_id: 0}
        self.passed_train_id = {origin_platform_id: []}
        for bd_platform in self.transfer_to_platform_list:
            self.left_behind_times[bd_platform] = 0
            self.passed_train_id.update({bd_platform: []})



def generate_event_list(events):
    event_list = []
    events = events.sort_values(['event_timestamp'])
    for event_id, line_id, direction_id, station_id, platform_id,platform_seq_no, train_id, event_timestamp, event_type in zip(
            events['event_id'], events['line_id'], events['direction_id'], events['station_id'],
            events['platform_id'], events['platform_seq_no'], events['train_id'],
            events['event_timestamp'], events['event_type']):
        event_list.append(Event(line_id, direction_id, station_id, platform_id,platform_seq_no, train_id, event_timestamp, event_type, event_id))
    return event_list


def offload_passengers(event, all_logs, all_trains, all_platforms):
    train = all_trains[event.train_id]
    platform = all_platforms[event.platform_id]

    all_logs['train_load_log'].append({
        'train_id': train.train_id,
        'timestamp': event.event_timestamp,
        'train_load_type': 'Arrival',
        'train_load': len(train.passenger_list),
        'platform_id': platform.platform_id,
    })

    all_logs['platform_queue_log'].append({
        'platform_id': platform.platform_id,
        'timestamp': event.event_timestamp,
        'queue_length': len(platform.passenger_list),
        'queue_length_type': 'When Arrival',
        'train_id': train.train_id,
    })

    remained_passengers = []
    remained_passengers_groupby_bd_station = OrderedDict() # already ordered
    for p in train.passenger_list:
        if len(p.transfer_from_platform_list) and platform.platform_id == p.transfer_from_platform_list[0]:
            # transfer
            p.trajectory['trajectory_type'].append('Transfer')
            p.trajectory['trajectory_time'].append(event.event_timestamp + _constant.DEFAULT_EXIT_WALKING_TIME)
            # next transfer platform
            p.transfer_from_platform_list.pop(0)
            next_platform_id = p.transfer_to_platform_list.pop(0)
            p.trajectory['trajectory_platform'].append(next_platform_id)
            p.trajectory['trajectory_event_id'].append(event.event_id)
            next_platform = all_platforms[next_platform_id]
            next_platform.passenger_list.append(p)
        elif p.destination == platform.platform_station_line_id:
            # leaving passenger
            p.trajectory['trajectory_platform'].append(platform.platform_id)
            p.trajectory['trajectory_type'].append('Exit')
            p.trajectory['trajectory_time'].append(event.event_timestamp + _constant.DEFAULT_EXIT_WALKING_TIME)
            p.trajectory['trajectory_event_id'].append(event.event_id)
            if 'completed_passenger_objects' in all_logs:
                all_logs['completed_passenger_objects'][p.passenger_id] = p
            # log passenger travel time
            all_logs['trajectory_log']['passenger_id'] += [p.passenger_id] * len(p.trajectory['trajectory_type'])
            all_logs['trajectory_log']['trajectory_type'] += p.trajectory['trajectory_type']
            all_logs['trajectory_log']['trajectory_time'] += p.trajectory['trajectory_time']
            all_logs['trajectory_log']['trajectory_platform'] += p.trajectory['trajectory_platform']


        else:
            remained_passengers.append(p)
            if p.trajectory['trajectory_platform'][-1] not in remained_passengers_groupby_bd_station:
                remained_passengers_groupby_bd_station[p.trajectory['trajectory_platform'][-1]] = []
            remained_passengers_groupby_bd_station[p.trajectory['trajectory_platform'][-1]].append(p)

    train.passenger_list = copy.deepcopy(remained_passengers)
    train.remaining_passenger_at_each_platform[platform.platform_id] = remained_passengers_groupby_bd_station
    return all_logs, all_trains, all_platforms

def get_num_board_passengers(available_space, platform, control_board_numbers = None, control_factor=None):
    if control_factor is None and control_board_numbers is None:
        num_onboard_pax = int(min(available_space, len(platform.passenger_list)))
    elif control_board_numbers is not None:
        num_onboard_pax = int(min(available_space, int(control_board_numbers), len(platform.passenger_list))) #
    elif control_factor is not None:
        num_onboard_pax = int(min(available_space, len(platform.passenger_list) * control_factor))
    else:
        raise Exception('Control board numbers not specified')
    return num_onboard_pax


def onboard_passengers(event, all_logs, all_trains, all_platforms, control_board_numbers=None, control_factor=None):
    train = all_trains[event.train_id]
    platform = all_platforms[event.platform_id]

    all_logs['platform_queue_log'].append({
        'platform_id': platform.platform_id,
        'timestamp': event.event_timestamp,
        'queue_length': len(platform.passenger_list),
        'queue_length_type': 'Before Onboard',
        'train_id': train.train_id,
        'control_log': f'control_board_numbers: {control_board_numbers}, control_factor: {control_factor}'
    })

    available_space = int(train.capacity) - len(train.passenger_list)
    num_onboard_pax = get_num_board_passengers(available_space, platform, control_board_numbers, control_factor)


    try:
        assert (num_onboard_pax >= 0) and (num_onboard_pax <= len(platform.passenger_list)) and (num_onboard_pax <= available_space)
    except AssertionError:
        print(f'platform {event.platform_id}, train_id {train.train_id}')
        print(f'wish to onboard: {num_onboard_pax}')
        print(f'queue length: {len(platform.passenger_list)}')
        print(f'available_space: {available_space}')
        print(f'control_board_numbers: {control_board_numbers}')
        print(f'control_factor: {control_factor}')
        exit()
    if len(platform.passenger_list) == 0:
        num_onboard_pax = 0

    to_board_passengers = platform.passenger_list[:num_onboard_pax]
    left_behind_passengers = platform.passenger_list[num_onboard_pax:]
    to_board_passenger_arrival_time_list = []
    for seq, p in enumerate(to_board_passengers):
        # ensure we have arrival / transfer events first
        assert p.trajectory['trajectory_platform'][-1] == platform.platform_id
        p.trajectory['trajectory_type'].append('Boarding')
        p.trajectory['trajectory_time'].append(event.event_timestamp)
        p.trajectory['trajectory_platform'].append(platform.platform_id)
        p.trajectory['trajectory_event_id'].append(event.event_id)
        p.boarding_seq_records[platform.platform_id] = seq
        all_logs['left_behind_log'].append({
            'passenger_id': p.passenger_id,
            'boarding_platform': platform.platform_id,
            'left_behind_times': p.left_behind_times[platform.platform_id],
            'previous_train_id': p.passed_train_id[platform.platform_id][-1] if len(p.passed_train_id[platform.platform_id]) > 0 else None,
            'boarded_train_id': train.train_id,
            'boarded_platform_seq': platform.platform_seq,
            'arrival_time_at_platform': p.trajectory['trajectory_time'][-1],
            'seq_at_queue_when_board': seq,
        })
        all_logs['left_behind_log_temp'][p.passenger_id] = {
            'passenger_id': p.passenger_id,
            'boarding_platform': platform.platform_id,
            'left_behind_times': p.left_behind_times[platform.platform_id],
            'previous_train_id': p.passed_train_id[platform.platform_id][-1] if len(p.passed_train_id[platform.platform_id]) > 0 else None,
            'boarded_train_id': train.train_id,
            'boarded_platform_seq': platform.platform_seq,
            'arrival_time_at_platform': p.trajectory['trajectory_time'][-1],
            'seq_at_queue_when_board': seq,
        }
        to_board_passenger_arrival_time_list.append(p.trajectory['trajectory_time'][-1])


    for p in left_behind_passengers:
        p.left_behind_times[platform.platform_id] += 1
        p.passed_train_id[platform.platform_id].append(train.train_id)
        all_logs['left_behind_log_temp'][p.passenger_id] = {
            'passenger_id': p.passenger_id,
            'boarding_platform': platform.platform_id,
            'left_behind_times': p.left_behind_times[platform.platform_id],
            'previous_train_id': p.passed_train_id[platform.platform_id][-1] if len(p.passed_train_id[platform.platform_id]) > 0 else None,
            'boarded_train_id': None,
            'boarded_platform_seq': platform.platform_seq,
            'arrival_time_at_platform': p.trajectory['trajectory_time'][-1],
            'seq_at_queue_when_board': None,
        }


    train.passenger_list.extend(to_board_passengers)
    train.available_capacity_at_each_platform_departure[platform.platform_id] = train.capacity - len(train.passenger_list)
    # update platform passengers
    platform.passenger_list = left_behind_passengers

    # if train.train_id == '2_0_17' and event.platform_id == '208_2_0':
    #     print('1')
    #     print(int(train.capacity), len(train.passenger_list))
    #     print(available_space, num_onboard_pax, len(platform.passenger_list))
    #     print([p.passenger_id for p in train.passenger_list])
    #     print([p.passenger_id for p in to_board_passengers])
    #     print([p.passenger_id for p in left_behind_passengers])
    # if train.train_id == '2_0_17' and event.platform_id == '210_2_0':
    #     print('2')
    #     print(int(train.capacity), len(train.passenger_list))
    #     print(available_space, num_onboard_pax, len(platform.passenger_list))
    #     print([p.passenger_id for p in train.passenger_list])
    #     print([p.passenger_id for p in to_board_passengers])
    #     print([p.passenger_id for p in left_behind_passengers])


    # Log train load and platform queue
    all_logs['train_load_log'].append({
        'train_id': train.train_id,
        'timestamp': event.event_timestamp,
        'train_load_type': 'Departure',
        'train_load': len(train.passenger_list),
        'platform_id': platform.platform_id,
        'control_log': f'control_board_numbers: {control_board_numbers}, control_factor: {control_factor}'
    })

    all_logs['platform_queue_log'].append({
        'platform_id': platform.platform_id,
        'timestamp': event.event_timestamp,
        'queue_length': len(platform.passenger_list),
        'queue_length_type': 'After Onboard',
        'train_id': train.train_id,
        'control_log': f'control_board_numbers: {control_board_numbers}, control_factor: {control_factor}'
    })

    if 'train_boarding_log' in all_logs:
        if train.train_id not in all_logs['train_boarding_log']:
            all_logs['train_boarding_log'][train.train_id] = {}
        all_logs['train_boarding_log'][train.train_id][platform.platform_id]= {
            'to_board_passenger_arrival_time_list': to_board_passenger_arrival_time_list
        } # already sorted

    return num_onboard_pax, all_logs, all_trains, all_platforms

def add_new_passengers_to_platform(event, all_platforms, grouped_passengers, pax_path_dict, passenger_objects):
    platform_id = event.platform_id
    platform = all_platforms[platform_id]

    start_time = platform.last_train_departure_time
    end_time = event.event_timestamp
    station_id = event.station_id

    pax_group = grouped_passengers.get(platform_id, None)
    if pax_group is None:
        return all_platforms, passenger_objects

    if isinstance(pax_group, dict) and 'passenger_id' in pax_group:
        pid_list = pax_group['passenger_id']
        tap_list = pax_group['tap_in_timestamp']
        dest_list = pax_group['destination_station_id']

        tap_arr = np.array(tap_list)
        i = int(tap_arr.searchsorted(start_time, side="left"))
        j = int(tap_arr.searchsorted(end_time, side="left"))
        if i >= j:
            return all_platforms, passenger_objects

        for k in range(i, j):
            pid = int(pid_list[k])
            dest = int(dest_list[k])
            tap_ts = int(tap_list[k])
            p = Passenger(station_id, dest, pid, platform.platform_id, pax_path_dict)
            p.tap_in_timestamp = tap_ts
            p.trajectory['trajectory_platform'].append(platform.platform_id)
            p.trajectory['trajectory_type'].append('Arrival')
            p.trajectory['trajectory_time'].append(event.event_timestamp)
            p.trajectory['trajectory_event_id'].append(event.event_id)
            passenger_objects[pid] = p
            platform.add_passenger(p)

        # update it to reduce search time next time
        pax_group['passenger_id'] = pid_list[j:]
        pax_group['tap_in_timestamp'] = tap_list[j:]
        pax_group['destination_station_id'] = dest_list[j:]
        grouped_passengers[station_id] = pax_group

    return all_platforms, passenger_objects



def initialize_trains(event, all_trains, train_capacity_dict):
    if event.train_id not in all_trains:
        all_trains[event.train_id] = Train(event.train_id, event.line_id, event.direction_id, train_capacity_dict)


def initialize_platforms(event_list, all_platforms, SIMULATION_START_TIMESTAMP):
    platform_line_idx = {}
    for event in event_list:
        if event.platform_id not in all_platforms:
            all_platforms[event.platform_id] = Platform(event.platform_id, event.line_id, event.direction_id, event.platform_seq_no, SIMULATION_START_TIMESTAMP)
            if (event.line_id, event.direction_id) not in platform_line_idx:
                platform_line_idx[(event.line_id, event.direction_id)] = []
            platform_line_idx[(event.line_id, event.direction_id)].append((event.platform_id, event.platform_seq_no))
    for key in platform_line_idx:
        platform_line_idx[key].sort(key=lambda x: x[1])
    return all_platforms, platform_line_idx
#
# def simulation_main(event_list):
#     initialize_platforms(event_list, all_platforms, SIMULATION_START_TIMESTAMP)
#     s_time = time.time()
#     for event_id, event in enumerate(event_list):
#         if event_id > 0 and event_id % 1000 == 0:
#             print('Start simulation event #{}, total {}'.format(event_id, len(event_list)))
#             total_spent_time = time.time() - s_time
#             estimate_total_finish_time = len(event_list) * (total_spent_time / event_id)
#             print(f'estimate_total_finish_time: {round(estimate_total_finish_time)} sec')
#         initialize_trains(event, all_trains, train_capacity_dict)
#
#
#         if event.event_type == 'Arrival':
#             offload_passengers(event, all_logs,  all_trains, all_platforms)
#         elif event.event_type == 'Departure':
#             add_new_passengers_to_platform(event, all_platforms, grouped_passengers, pax_path_dict, passenger_objects)
#             onboard_passengers(event, all_logs,  all_trains, all_platforms)
#
#     print('total spent time: {} sec'.format(round(time.time() - s_time)))
#
#     # Save output metrics
#     # os.makedirs('data', exist_ok=True)
#     # pd.DataFrame(train_load_log).to_csv('data/train_load.csv', index=False)
#     # pd.DataFrame(platform_queue_log).to_csv('data/platform_queue.csv', index=False)
#     # pd.DataFrame(left_behind_log).to_csv('data/left_behind.csv', index=False)
#     # pd.DataFrame(passenger_travel_log).to_csv('data/passenger_travel_times.csv', index=False)
#     print("Success")


def process_passenger_group_by_origin(passenger_df_path_raw):
    passenger_df_path = passenger_df_path_raw.copy()
    passenger_df_path = passenger_df_path.sort_values("tap_in_timestamp")
    all_passengers = (
        passenger_df_path.groupby(['origin_platform_id'])[['passenger_id', 'destination_station_id', 'tap_in_timestamp']]
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
        origin_platform_id=('from_platform_id', 'first'),
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

    return pax_path_dict, passenger_df_path


def save_all_logs(all_logs):
    trajectory_log_df = pd.DataFrame(all_logs['trajectory_log'])
    trajectory_log_df.to_csv('output/trajectory_log.csv', index=False)

    train_log_df = pd.DataFrame(all_logs['train_load_log'])
    train_log_df.to_csv('output/train_log.csv', index=False)

    queue_length_log_df = pd.DataFrame(all_logs['platform_queue_log'])
    queue_length_log_df.to_csv('output/platform_queue_log.csv', index=False)


    left_behind_log_df = pd.DataFrame(all_logs['left_behind_log'])
    left_behind_log_df.to_csv('output/left_behind_log.csv', index=False)



# if __name__ == '__main__':
#     # train_load_log = []  # Each departure: train_id, timestamp, load
#     # platform_queue_log = []  # At each event: platform_id, timestamp, queue_length
#     # left_behind_log = []  # Track passengers who could not board train
#     # passenger_travel_log = []  # For each passenger: id, origin, destination, boarding_time, alighting_time, total_travel_time
#
#     all_logs = {
#         'trajectory_log': {'passenger_id': [], 'trajectory_type': [], 'trajectory_time': [], 'trajectory_platform': []},
#         'train_load_log': [],
#         'platform_queue_log': [],
#         'left_behind_log': [],
#         'left_behind_log_temp': {},
#     }
#
#
#     SIMULATION_START_TIMESTAMP = 6 * 3600
#     SIMULATION_END_TIMESTAMP = 10 * 3600
#
#     train_capacity_df = pd.read_csv('data/train_capacity.csv')
#     train_capacity_df['train_capacity'] = np.round(train_capacity_df['train_capacity'] * _constant.TRAIN_CAPACITY_FACTOR)
#     train_capacity_df['train_capacity'] = train_capacity_df['train_capacity'].astype('int')
#     train_capacity_dict = train_capacity_df.set_index(['line_id', 'direction_id'])['train_capacity'].to_dict()
#
#     passenger_df = pd.read_csv('data/individual_demands.csv')
#     passenger_df = passenger_df.loc[
#         (passenger_df['tap_in_timestamp'] > SIMULATION_START_TIMESTAMP) &
#         (passenger_df['tap_in_timestamp'] < SIMULATION_END_TIMESTAMP)
#     ]
#
#
#
#     path_df = pd.read_csv('data/paths.csv')
#     pax_path_dict = assign_passenger_path(passenger_df, path_df)
#     print('Finish assigning passenger paths...')
#
#     grouped_passengers = process_passenger_group_by_origin(passenger_df)
#     passenger_objects = {}
#
#     all_trains = {}
#     all_platforms = {}
#
#     events = pd.read_csv('data/events.csv')
#     events = events.loc[
#         (events['event_timestamp'] > SIMULATION_START_TIMESTAMP) &
#         (events['event_timestamp'] < SIMULATION_END_TIMESTAMP)
#     ]
#
#     event_list = generate_event_list(events)
#     print('Finish generating event list...')
#     simulation_main(event_list)
#     print('Finish simulating event list, start to save logs...')
#     save_all_logs()
