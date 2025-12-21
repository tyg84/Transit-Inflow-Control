import pandas as pd
import numpy as np
import _constant
import time
from B01_simulation import (process_passenger_group_by_origin,
                            assign_passenger_path, generate_event_list, save_all_logs,
                            initialize_platforms, initialize_trains, offload_passengers,
                            add_new_passengers_to_platform, get_num_board_passengers,
                            onboard_passengers
                            )


def simulation_with_control(event_list, all_logs, iteration, control_factor_dict):

    initialize_platforms(event_list, all_platforms, SIMULATION_START_TIMESTAMP)
    s_time = time.time()
    for event_id, event in enumerate(event_list):
        if event_id > 0 and event_id % 1000 == 0:
            print('Start simulation event #{}, total {}'.format(event_id, len(event_list)))
            total_spent_time = time.time() - s_time
            estimate_total_finish_time = len(event_list) * (total_spent_time / event_id)
            print(f'estimate_total_finish_time: {round(estimate_total_finish_time)} sec')
        initialize_trains(event, all_trains, train_capacity_dict)

        if event.event_type == 'Arrival':
            offload_passengers(event, all_logs, all_trains, all_platforms)
        elif event.event_type == 'Departure':
            add_new_passengers_to_platform(event, all_platforms, grouped_passengers, pax_path_dict, passenger_objects)
            control_factor = control_factor_dict.get(event.event_id)
            if control_factor is None and iteration >= 1:
                raise Exception('Control factor cannot be None after 1st iteration')
            if control_factor is None:
                control_factor_temp = onboard_passengers(event, all_logs, all_trains, all_platforms, control_factor)
                control_factor_dict[event.event_id] = control_factor_temp
            else:
                onboard_passengers(event, all_logs, all_trains, all_platforms, control_factor)

    print('total spent time: {} sec'.format(round(time.time() - s_time)))

    # Save output metrics
    # os.makedirs('data', exist_ok=True)
    # pd.DataFrame(train_load_log).to_csv('data/train_load.csv', index=False)
    # pd.DataFrame(platform_queue_log).to_csv('data/platform_queue.csv', index=False)
    # pd.DataFrame(left_behind_log).to_csv('data/left_behind.csv', index=False)
    # pd.DataFrame(passenger_travel_log).to_csv('data/passenger_travel_times.csv', index=False)
    print("Success")
    return control_factor_dict



def save_all_logs_with_iteration(all_logs, iteration):
    trajectory_log_df = pd.DataFrame(all_logs['trajectory_log'])
    trajectory_log_df.to_csv(f'output/trajectory_log_iteration_{iteration}.csv', index=False)

    train_log_df = pd.DataFrame(all_logs['train_load_log'])
    train_log_df.to_csv(f'output/train_log_iteration_{iteration}.csv', index=False)

    queue_length_log_df = pd.DataFrame(all_logs['platform_queue_log'])
    queue_length_log_df.to_csv(f'output/platform_queue_log_iteration_{iteration}.csv', index=False)


    left_behind_log_df = pd.DataFrame(all_logs['left_behind_log'])
    left_behind_log_df.to_csv(f'output/left_behind_log_iteration_{iteration}.csv', index=False)

def update_control_strategy(all_logs, control_factor_dict):
    lb_log_df = pd.DataFrame(all_logs['left_behind_log'])
    max_lb_pax_times = lb_log_df['left_behind_times'].max()
    max_lb_pax = lb_log_df.loc[lb_log_df['left_behind_times'] == max_lb_pax_times]
    print(f'lb_pax: {len(max_lb_pax)}', f'max_lb_pax_times: {max_lb_pax_times}')

    IF_UPDATE = False
    for boarded_train_id, pax_group in max_lb_pax.groupby(['boarded_train_id']):
        platform_id = pax_group['boarding_platform'].iloc[0]
        # to_board_passenger_seq_at_queue = all_logs['train_boarding_log'][train_id][platform_id]['board_passenger_seq_at_queue']
        reserved_space = np.max(pax_group['seq_at_queue_when_board']) + 1
        print(f'boarded_train_id: {boarded_train_id}, reserved_space: {reserved_space}')
        #### next step: get previous train, try to reserve reserved_space seats, by remove passengers from upstream stations.
        p_lb = pax_group['passenger_id'].iloc[0]
        previous_train_id = passenger_objects[p_lb].passed_train_id[platform_id][-1]
        previous_train = all_trains[previous_train_id]
        remained_pax_dict = previous_train.remaining_passenger_at_each_platform[platform_id]
        removed_pax_cnt = 0
        to_remove_pax_by_upstream_station = {}
        for LB_time in range(max_lb_pax_times):
            for plat_id, remained_pax in remained_pax_dict.items():
                remained_pax = remained_pax_dict[plat_id]
                for p_ob in remained_pax[::-1]: # reverse order, starting from last pax in queue
                    if plat_id not in to_remove_pax_by_upstream_station:
                        to_remove_pax_by_upstream_station[plat_id] = []
                    if p_ob.left_behind_times[plat_id] == LB_time:
                        to_remove_pax_by_upstream_station[plat_id].append(p_ob)
                        removed_pax_cnt += 1
                    if removed_pax_cnt == reserved_space:
                        for to_remove_platform, p_list in to_remove_pax_by_upstream_station.items():
                            control_event_id = p_list[0].trajectory['trajectory_event_id'][-1] # boarding event id
                            old_num_board = control_factor_dict[control_event_id]
                            control_factor_dict[control_event_id] = old_num_board - len(p_list)
                            print(f'Update Event: {control_event_id} onboard pax from {old_num_board} to {control_factor_dict[control_event_id]}')
                            IF_UPDATE = True
    if IF_UPDATE:
        return control_factor_dict
    else:
        print('No update can be performed')
        return None


if __name__ == '__main__':

    SIMULATION_START_TIMESTAMP = 6 * 3600
    SIMULATION_END_TIMESTAMP = 8 * 3600

    train_capacity_df = pd.read_csv('data/train_capacity.csv')
    train_capacity_df['train_capacity'] = np.round(
        train_capacity_df['train_capacity'] * _constant.TRAIN_CAPACITY_FACTOR)
    train_capacity_df['train_capacity'] = train_capacity_df['train_capacity'].astype('int')
    train_capacity_dict = train_capacity_df.set_index(['line_id', 'direction_id'])['train_capacity'].to_dict()

    passenger_df = pd.read_csv('data/individual_demands.csv')
    passenger_df = passenger_df.loc[
        (passenger_df['tap_in_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (passenger_df['tap_in_timestamp'] < SIMULATION_END_TIMESTAMP)
        ]



    path_df = pd.read_csv('data/paths.csv')
    pax_path_dict = assign_passenger_path(passenger_df, path_df)
    print('Finish assigning passenger paths...')


    events = pd.read_csv('data/events.csv')
    events = events.loc[
        (events['event_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (events['event_timestamp'] < SIMULATION_END_TIMESTAMP)
        ]

    event_list = generate_event_list(events)
    print('Finish generating event list...')
    control_factor_dict = {}
    for iteration in range(5):
        print('=====Iteration {}======'.format(iteration))
        all_trains = {}
        all_platforms = {}
        passenger_objects = {}
        all_logs = {
            'trajectory_log': {'passenger_id': [], 'trajectory_type': [], 'trajectory_time': [],
                               'trajectory_platform': []},
            'train_load_log': [],
            'platform_queue_log': [],
            'left_behind_log': [],
            'train_boarding_log': {}
        }
        grouped_passengers = process_passenger_group_by_origin(passenger_df)


        control_factor_dict = simulation_with_control(event_list, all_logs, iteration, control_factor_dict)
        control_factor_dict = update_control_strategy(all_logs, control_factor_dict)
        print('Finish simulating event list, start to save logs...')
        save_all_logs_with_iteration(all_logs, iteration)