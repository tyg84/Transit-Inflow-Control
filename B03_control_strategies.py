import pandas as pd
import numpy as np
import time
import os

from B01_simulation import (process_passenger_group_by_origin,
                            assign_passenger_path, generate_event_list, initialize_platforms, initialize_trains, offload_passengers,
                            add_new_passengers_to_platform, onboard_passengers
                            )
from collections import Counter


def simulation_with_control(
        event_list, passenger_objects, all_platforms, all_trains, all_logs, iteration,
        control_board_num_dict, control_factor_dict, grouped_passengers, BOARD_NUM_CONTROL):

    all_platforms, platform_line_index = initialize_platforms(event_list, all_platforms, SIMULATION_START_TIMESTAMP)
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
    return control_board_num_dict, control_factor_dict, passenger_objects, all_platforms, all_trains, all_logs, platform_line_index



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

def update_control_dict(to_remove_pax_by_upstream_station, platforms_to_keep_board_numbers, previous_train_id, control_factor_dict, platform_id, removed_pax_cnt, all_platforms, additional_board_increase_to_utilize_avail_cap):
    ## controlled platforms
    for to_remove_platform, p_list in to_remove_pax_by_upstream_station.items():
        if len(p_list) == 0:
            continue
        old_num_board = control_factor_dict[(previous_train_id, to_remove_platform)]['Onboard']

        first_board_p = p_list[-1]
        seq_at_queue = first_board_p.boarding_seq_records[to_remove_platform]
        new_num_board = seq_at_queue ## seq_at_queue start with 0

        # new_num_board = max(0, old_num_board - len(p_list))

        control_factor_dict[(previous_train_id, to_remove_platform)]['Control'] = new_num_board
        print(
            f'Update train: '
            f'{previous_train_id}, update platform: {to_remove_platform}, platform_seq: {all_platforms[to_remove_platform].platform_seq}, '
            f'onboard pax from {old_num_board} to {new_num_board}')
    ## keep board numbers:
    for plat_id in platforms_to_keep_board_numbers:
        old_num_board = control_factor_dict[(previous_train_id, plat_id)]['Onboard']
        control_factor_dict[(previous_train_id, plat_id)]['Control'] = old_num_board
        print(
            f'Keep old board, train: {previous_train_id}, platform: {plat_id}, seq: {all_platforms[plat_id].platform_seq}, onboard pax {old_num_board}')
    ## current platform
    if control_factor_dict[(previous_train_id, platform_id)]['Control'] is not None:
        control_factor_dict[(previous_train_id, platform_id)]['Control'] += removed_pax_cnt + additional_board_increase_to_utilize_avail_cap


    return control_factor_dict


def get_in_the_middle_platforms(platform_line_index, line_id, dir_id, seq_id_start, seq_id_end):
    all_platforms_in_line_dir = platform_line_index[(line_id, dir_id)]
    ## already sorted
    middle_platforms = all_platforms_in_line_dir[seq_id_start:seq_id_end-1]
    return [plat_seq[0] for plat_seq in middle_platforms]



def update_control_strategy(all_logs, all_trains, all_platforms, passenger_objects, control_factor_dict, platform_line_index, best_results, iteration):
    lb_log_df = pd.DataFrame(all_logs['left_behind_log'])
    max_lb_pax_times = lb_log_df['left_behind_times'].max()

    if max_lb_pax_times < best_results['best_max_LB']:
        best_results['best_max_LB'] = int(max_lb_pax_times)
        best_results['best_iteration'] = iteration

    if max_lb_pax_times <= 1:
        print(f'Max LB == {max_lb_pax_times}, No Control Needed')
        return {}, True, best_results
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
        #
        if (line_id, dir_id) in processed_line_dir:
            print(f'Line Dir {(line_id, dir_id)} already processed, skip it')
            continue

        processed_line_dir.append((line_id, dir_id))

        reserved_space = np.max(pax_group['seq_at_queue_when_board']) + 1
        print(f'boarded train {boarded_train_id}, platform: {platform_id}, platform seq: {all_platforms[platform_id].platform_seq}, reserved_space: {reserved_space}')
        #### next step: get previous train, try to reserve reserved_space seats, by remove passengers from upstream stations.

        previous_train = all_trains[previous_train_id]
        previous_train_available_capacity_when_departure = previous_train.available_capacity_at_each_platform_departure[platform_id]
        additional_board_increase_to_utilize_avail_cap = 0
        remained_pax_dict = previous_train.remaining_passenger_at_each_platform[platform_id]
        removed_pax_cnt = 0
        to_remove_pax_by_upstream_station = {}
        platforms_to_keep_board_all = []
        for prev_platform_id, remained_pax in list(remained_pax_dict.items())[::-1]: # reverse order, from prev 1
            for p_ob in remained_pax[::-1]: # reverse order, starting from last pax in queue
                if prev_platform_id not in to_remove_pax_by_upstream_station:
                    to_remove_pax_by_upstream_station[prev_platform_id] = []
                # if p_ob.left_behind_times[prev_platform_id] < max_lb_pax_times - 1:
                if all_platforms[prev_platform_id].platform_seq == 1:
                    # first platform form
                    # not allow it becomes worse
                    if p_ob.left_behind_times[prev_platform_id] >= max_lb_pax_times - 1:
                        continue

                to_remove_pax_by_upstream_station[prev_platform_id].append(p_ob)
                removed_pax_cnt += 1
                # else:
                #     print('reach max num passengers to control')
                #     control_factor_dict = update_control_dict(to_remove_pax_by_upstream_station, previous_train_id, control_factor_dict)
                #     break

                if removed_pax_cnt == reserved_space:
                    break
            print(f'process platform {prev_platform_id}, seq {all_platforms[prev_platform_id].platform_seq}, block pax: {len(to_remove_pax_by_upstream_station[prev_platform_id])}')
            if removed_pax_cnt == reserved_space:
                break
        if removed_pax_cnt == reserved_space:
            print('reach reserved space')
            control_factor_dict = update_control_dict(
                to_remove_pax_by_upstream_station, platforms_to_keep_board_all, previous_train_id, control_factor_dict, platform_id, removed_pax_cnt, all_platforms, additional_board_increase_to_utilize_avail_cap)
        else:
            print(f'Not enough reserved space. need {reserved_space}, reserved {removed_pax_cnt}')
            if previous_train_available_capacity_when_departure > 0:
                additional_board_increase_to_utilize_avail_cap = min(
                    reserved_space-removed_pax_cnt, previous_train_available_capacity_when_departure) ## use additional cap
                print(f'Add additional board increase {additional_board_increase_to_utilize_avail_cap} to utilize avail capacity')
            # if removed_pax_cnt == 0:
            #     print('No more passengers to block, exit')
            #     return {}, True, best_results
            control_factor_dict = update_control_dict(
                to_remove_pax_by_upstream_station, platforms_to_keep_board_all, previous_train_id, control_factor_dict, platform_id, removed_pax_cnt, all_platforms, additional_board_increase_to_utilize_avail_cap)

        # ### to as much as we can to avoid too many max LB
        # total_remove_pax = sum([len(v) for k,v in to_remove_pax_by_upstream_station.items()])
        # print(f'total remove pax if not reach: {total_remove_pax}')
        # if total_remove_pax > 0:
        #     control_factor_dict = update_control_dict(to_remove_pax_by_upstream_station, previous_train_id, control_factor_dict)


    return control_factor_dict, False, best_results




if __name__ == '__main__':


    case_name = 'reference'

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

    best_results = {'best_iteration': 0, 'best_max_LB': np.inf}

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


        control_board_num_dict, _, passenger_objects, all_platforms, all_trains, all_logs, platform_line_index = simulation_with_control(
            event_list, passenger_objects, all_platforms, all_trains,
            all_logs, iteration, control_board_num_dict,None, grouped_passengers, BOARD_NUM_CONTROL=True)
        control_board_num_dict, STOP, best_results = update_control_strategy(
            all_logs, all_trains, all_platforms, passenger_objects, control_board_num_dict, platform_line_index, best_results, iteration)
        print('Finish simulating event list, start to save logs...')
        save_all_logs_with_iteration(all_logs, iteration, case_name)

        print(f'**********Best Results: {best_results}***********')
        if STOP:
            break
