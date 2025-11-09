import pandas as pd
import numpy as np
import _constant
import copy

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
        self.origin = pax_path_dict[passenger_id]['origin']
        self.destination = pax_path_dict[passenger_id]['destination']
        self.passenger_id = passenger_id
        self.path_id = pax_path_dict[passenger_id]['path_id']
        self.transfer_from_station_list = pax_path_dict[passenger_id]['from_transfer_stations'].split(',')
        self.transfer_to_station_list = pax_path_dict[passenger_id]['to_transfer_stations'].split(',')
        self.origin_line_id = int(pax_path_dict[passenger_id]['origin'].split('_')[1])
        self.trajectory = {'trajectory_type': [], 'trajectory_time': []}







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
    ### offload passenger from the train if this is a transfer station or destination
    ## offload: move them out of the train, put to next boarding platform if transfer (event_time + transfer walk time),
    # or delete them out of the system when finish the trip record exit time (event time + walk time out).
    train = all_trains[event.train_id]
    if len(train.passenger_list) == 0:
        return
    platform = all_platforms[event.platform_id]

    remained_passengers = []

    for p in train.passenger_list:
        if len(p.transfer_from_station_list) and platform.platform_station_line_id in p.transfer_from_station_list:
            # Transfer
            p.trajectory['trajectory_type'].append('Transfer')
            p.trajectory['trajectory_time'].append(event.event_timestamp + _constant.DEFAULT_EXIT_WALKING_TIME)
            # add passenger to new platforms
            # FINSH THIS PART

        elif p.destination_station_id == platform.platform_station_line_id:
            p.trajectory['trajectory_type'].append('Exit')
            p.trajectory['trajectory_time'].append(event.event_timestamp + _constant.DEFAULT_EXIT_WALKING_TIME)
        else:
            remained_passengers.append(p)

    train.passenger_list = copy.deepcopy(remained_passengers)
    return


def onboard_passengers(event):
    ### put new passengers from entry gate to the platform. sort by arrival-at-platform time
    # onboard passenger to the train up to the capacity.
    ## update the platform queue.
    train = all_trains[event.train_id]
    platform = all_platforms[event.platform_id]
    available_space = train.capacity - len(train.passenger_list)
    if available_space == 0:
        return
    to_board_passengers = platform.passenger_list[:available_space]
    for p in to_board_passengers:
        p.trajectory['trajectory_type'].append('Boarding')
        p.trajectory['trajectory_time'].append(event.event_timestamp)
    ## add pax to train
    train.passenger_list.extend(to_board_passengers)
    ## delete pax from platform
    platform.passenger_list = platform.passenger_list[available_space:]
    return


def add_new_passengers_to_platform(event):
    platform_id = event.platform_id
    platform = all_platforms[platform_id]
    start_time = platform.last_train_departure_time
    end_time = event.event_timestamp
    station_id = event.station_id
    tap_in_time_list = np.array(all_passengers[station_id]['tap_in_timestamp'])
    i = tap_in_time_list.searchsorted(start_time, side="left")
    j = tap_in_time_list.searchsorted(end_time, side="left")
    pax_id_list = all_passengers[station_id]['passenger_id'][i:j]
    tap_int_time_list = all_passengers[station_id]['tap_in_timestamp'][i:j]
    des_list = all_passengers[station_id]['destination_station_id'][i:j]

    for passenger_id, origin_station_id, destination_station_id in zip(pax_id_list, tap_int_time_list, des_list):
        all_passengers[passenger_id] = Passenger(origin_station_id, destination_station_id, passenger_id)
        platform.add_passenger(all_passengers[passenger_id])
    # drop all passengers before i to accelerate future search
    all_passengers[station_id]['passenger_id'] = all_passengers[station_id]['passenger_id'][i:]
    all_passengers[station_id]['tap_in_timestamp'] = all_passengers[station_id]['tap_in_timestamp'][i:]
    all_passengers[station_id]['destination_station_id'] = all_passengers[station_id]['destination_station_id'][i:]


def initialize_trains(event):
    if event.train_id not in all_trains:
        all_trains[event.train_id] = Train(event.train_id, event.line_id, event.direction_id)
    return

def initialize_platforms(event):
    if event.platform_id not in all_platforms:
        all_platforms[event.platform_id] = Platform(event.platform_id, event.line_id, event.direction_id)
    return




def simulation_main(event_list):

    for event in event_list:
        print(f'Event: {event.event_timestamp}')
        initialize_trains(event)
        initialize_platforms(event)

        if event.event_type == 'Arrival':
            offload_passengers(event)
        elif event.event_type == 'Departure':
            add_new_passengers_to_platform(event)
            onboard_passengers(event)


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
    #

    unique_path = path_df.groupby(['origin_station_id', 'destination_station_id', 'origin', 'destination', 'path_id']).agg(
        total_travel_time=('cumulated_travel_time', 'last'),
        from_transfer_stations=('from_station', lambda x: ",".join(x[path_df.loc[x.index, 'direction_id'].eq(2)].unique())),
        to_transfer_stations=('to_station',
                                lambda x: ",".join(x[path_df.loc[x.index, 'direction_id'].eq(2)].unique()))
    ).reset_index()

    # # get total travel time
    # unique_path = path_df.groupby(['origin','destination', 'path_id']).last().reset_index()
    # # sort travel time
    unique_path = unique_path.sort_values(['total_travel_time'], ascending=True)
    unique_path = unique_path.groupby(['origin_station_id','destination_station_id', 'path_id']).first().reset_index()
    passenger_df_path = passenger_df.merge(unique_path, on=['origin_station_id', 'destination_station_id'])
    # check path num
    check_path_num = passenger_df_path.groupby(['passenger_id'])['path_id'].count().reset_index()
    check_path_num = check_path_num.loc[check_path_num['path_id']>1]
    assert len(check_path_num) == 0

    pax_path_dict = passenger_df_path[[
        'passenger_id','origin','destination','path_id','from_transfer_stations', 'to_transfer_stations'
    ]].set_index('passenger_id').to_dict(orient='index')

    return pax_path_dict



if __name__ == '__main__':
    SIMULATION_START_TIMESTAMP = 5 * 3600
    SIMULATION_END_TIMESTAMP = 24 * 3600
    ## generate demand first.
    train_capacity_df = pd.read_csv('data/train_capacity.csv')
    train_capacity_dict = train_capacity_df.set_index(['line_id', 'direction_id'])['train_capacity'].to_dict()
    passenger_df = pd.read_csv('data/individual_demands.csv')
    passenger_df = passenger_df.loc[
        (passenger_df['tap_in_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (passenger_df['tap_in_timestamp'] < SIMULATION_END_TIMESTAMP)
    ]
    all_passengers = process_passenger_group_by_origin(passenger_df)
    path_df = pd.read_csv('data/paths.csv')
    pax_path_dict = assign_passenger_path(passenger_df, path_df)

    all_trains = {}
    train_capacity = {}
    all_platforms = {}
    events = pd.read_csv('data/events.csv')
    events = events.loc[
        (events['event_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (events['event_timestamp'] < SIMULATION_END_TIMESTAMP)
    ]
    event_list = generate_event_list(events)
    simulation_main(event_list)