import pandas as pd
import numpy as np
import time
import os

from A05_generate_train_capacity import train_capacity_df
from B01_simulation import (process_passenger_group_by_origin,
                            assign_passenger_path, generate_event_list, initialize_platforms, initialize_trains, offload_passengers,
                            add_new_passengers_to_platform, onboard_passengers
                            )
from collections import Counter


def simulation_with_control(
        event_list, passenger_objects, all_platforms, all_trains, all_logs, iteration,
        control_board_num_dict, control_factor_dict, grouped_passengers, BOARD_NUM_CONTROL):

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
            all_logs, all_trains, all_platforms = offload_passengers(event, all_logs, all_trains, all_platforms)
        elif event.event_type == 'Departure':
            if BOARD_NUM_CONTROL:
                all_platforms, passenger_objects = add_new_passengers_to_platform(event, all_platforms, grouped_passengers, pax_path_dict, passenger_objects)
                control_board_num = control_board_num_dict.get(
                    (event.train_id, event.platform_id), {'Control': None, 'Onboard': None})
                if control_board_num['Control'] is None:
                    num_ob_pax, all_logs, all_trains, all_platforms = onboard_passengers(event, all_logs, all_trains, all_platforms, control_board_num['Control'], control_factor=None)
                    control_board_num_dict[(event.train_id, event.platform_id)] = {'Control': None, 'Onboard': num_ob_pax}
                else:
                    num_ob_pax, all_logs, all_trains, all_platforms = onboard_passengers(event, all_logs, all_trains, all_platforms, control_board_num['Control'], control_factor=None)
                    control_board_num_dict[(event.train_id, event.platform_id)]['Onboard'] = num_ob_pax
            else:
                all_platforms, passenger_objects = add_new_passengers_to_platform(event, all_platforms, grouped_passengers, pax_path_dict, passenger_objects)
                control_factor = control_factor_dict.get((event.train_id, event.platform_id))
                num_ob_pax, all_logs, all_trains, all_platforms = onboard_passengers(
                    event, all_logs, all_trains, all_platforms, control_board_numbers=None, control_factor=control_factor)

    print('total spent time: {} sec'.format(round(time.time() - s_time)))

    # Save output metrics
    # os.makedirs('data', exist_ok=True)
    # pd.DataFrame(train_load_log).to_csv('data/train_load.csv', index=False)
    # pd.DataFrame(platform_queue_log).to_csv('data/platform_queue.csv', index=False)
    # pd.DataFrame(left_behind_log).to_csv('data/left_behind.csv', index=False)
    # pd.DataFrame(passenger_travel_log).to_csv('data/passenger_travel_times.csv', index=False)
    print("Success")
    return control_board_num_dict, control_factor_dict, passenger_objects, all_platforms, all_trains, all_logs



def save_all_logs_with_iteration(all_logs, iteration, case_name):
    os.makedirs(f'output/{case_name}', exist_ok=True)

    trajectory_log_df = pd.DataFrame(all_logs['trajectory_log'])
    trajectory_log_df.to_csv(f'output/{case_name}/trajectory_log_iteration_{iteration}.csv', index=False)

    train_log_df = pd.DataFrame(all_logs['train_load_log'])
    train_log_df.to_csv(f'output/{case_name}/train_log_iteration_{iteration}.csv', index=False)

    queue_length_log_df = pd.DataFrame(all_logs['platform_queue_log'])
    queue_length_log_df.to_csv(f'output/{case_name}/platform_queue_log_iteration_{iteration}.csv', index=False)


    left_behind_log_df = pd.DataFrame(all_logs['left_behind_log'])
    left_behind_log_df.to_csv(f'output/{case_name}/left_behind_log_iteration_{iteration}.csv', index=False)



def update_control_strategy(all_logs, all_trains, all_platforms, passenger_objects, control_factor_dict):
    lb_log_df = pd.DataFrame(all_logs['left_behind_log'])
    max_lb_pax_times = lb_log_df['left_behind_times'].max()
    if max_lb_pax_times <= 1:
        print(f'Max LB == {max_lb_pax_times}, No Control Needed')
        exit()
    # lb_threshold = 2
    max_lb_pax = lb_log_df.loc[lb_log_df['left_behind_times'] == max_lb_pax_times].copy()
    lb_patterns = dict(Counter(max_lb_pax['left_behind_times']))
    # objective = sum([k**2 * v for k,v in lb_patterns.items()]) ## LB^2 * num people

    print('LB patterns:', lb_patterns)
    # print('***Objective***:', objective)
    # print(f'lb_pax: {len(max_lb_pax)}', f'max_lb_pax_times: {max_lb_pax_times}')
    max_lb_pax = max_lb_pax.sort_values(['boarded_train_id','boarded_platform_seq'], ascending=[True, False])
    processed_line_dir = []

    for boarded_train_id, pax_group in max_lb_pax.groupby('boarded_train_id'):
        line_id = int(boarded_train_id.split('_')[0])
        dir_id = int(boarded_train_id.split('_')[1])
        previous_train_id = pax_group['previous_train_id'].iloc[0]
        platform_id = pax_group['boarding_platform'].iloc[0]

        #     print(f'Train platform {(previous_train_id, platform_id)} is already controlled, cannot further reduce LB')
        #     continue

        # if (line_id, dir_id) in processed_line_dir:
        #     print(f'Line Dir {(line_id, dir_id)} already processed, skip it')
        #     continue

        processed_line_dir.append((line_id, dir_id))

        reserved_space = np.max(pax_group['seq_at_queue_when_board']) + 1
        print(f'boarded train and platform: {boarded_train_id, platform_id, all_platforms[platform_id].platform_seq}, reserved_space: {reserved_space}')
        #### next step: get previous train, try to reserve reserved_space seats, by remove passengers from upstream stations.

        previous_train = all_trains[previous_train_id]

        marginal_reduction = 0.05

        passed_platform = list(previous_train.remaining_passenger_at_each_platform.keys())
        passed_platform_id_and_seq = [(plat_id, all_platforms[plat_id].platform_seq) for plat_id in passed_platform if all_platforms[plat_id].platform_seq < all_platforms[platform_id].platform_seq]
        passed_platform_id_and_seq = sorted(passed_platform_id_and_seq, key=lambda x: -x[1])
        cum_remove = 0
        for to_remove_platform, seq in passed_platform_id_and_seq:
            old_num_board = control_factor_dict[(previous_train_id, to_remove_platform)]['Onboard']
            if old_num_board > 0:
                reduct_pax = max(1, int(old_num_board * marginal_reduction))
                new_num_board = old_num_board - reduct_pax
            else:
                reduct_pax = 0
                new_num_board = old_num_board ## do not reduce
            control_factor_dict[(previous_train_id, to_remove_platform)]['Control'] = new_num_board
            print(
                f'Update Control: {(previous_train_id, to_remove_platform, all_platforms[to_remove_platform].platform_seq)} onboard pax from {old_num_board} to {new_num_board}')
            cum_remove += reduct_pax
            if cum_remove >= reserved_space:
                break
        ## current platform
        if control_factor_dict[(previous_train_id, platform_id)]['Control'] is not None:
            control_factor_dict[(previous_train_id, platform_id)]['Control'] += cum_remove

    return control_factor_dict




if __name__ == '__main__':


    case_name = 'equity_efficiency_MLR'

    SIMULATION_START_TIMESTAMP = 6 * 3600
    SIMULATION_END_TIMESTAMP = 9 * 3600

    train_capacity_df = pd.read_csv(f'data/{case_name}/train_capacity_adjusted.csv')
    train_capacity_dict = train_capacity_df.set_index(['line_id', 'direction_id'])['train_capacity'].to_dict()

    passenger_df = pd.read_csv(f'data/{case_name}/individual_demands.csv')
    passenger_df = passenger_df.loc[
        (passenger_df['tap_in_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (passenger_df['tap_in_timestamp'] < SIMULATION_END_TIMESTAMP)
        ]



    path_df = pd.read_csv(f'data/{case_name}/paths.csv')
    pax_path_dict, passenger_df_path = assign_passenger_path(passenger_df, path_df)
    print('Finish assigning passenger paths...')


    events = pd.read_csv(f'data/{case_name}/events.csv')
    events = events.loc[
        (events['event_timestamp'] > SIMULATION_START_TIMESTAMP) &
        (events['event_timestamp'] < SIMULATION_END_TIMESTAMP)
        ]

    event_list = generate_event_list(events)
    print('Finish generating event list...')
    control_board_num_dict = {}

    for iteration in range(100):
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
            'train_boarding_log': {},
            'left_behind_log_temp': {} # keyed by pax id, get real time lb info
        }
        grouped_passengers = process_passenger_group_by_origin(passenger_df_path)


        control_board_num_dict, _, passenger_objects, all_platforms, all_trains, all_logs = simulation_with_control(
            event_list, passenger_objects, all_platforms, all_trains,
            all_logs, iteration, control_board_num_dict,None, grouped_passengers, BOARD_NUM_CONTROL=True)
        control_board_num_dict = update_control_strategy(
            all_logs, all_trains, all_platforms, passenger_objects, control_board_num_dict)
        print('Finish simulating event list, start to save logs...')
        save_all_logs_with_iteration(all_logs, iteration, case_name)
