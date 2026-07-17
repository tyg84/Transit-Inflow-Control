
import pandas as pd
import numpy as np
import time

from B01_simulation import (process_passenger_group_by_origin,
                            assign_passenger_path, generate_event_list, initialize_platforms, initialize_trains, offload_passengers,
                            add_new_passengers_to_platform, onboard_passengers
                            )
from B03_control_strategies import save_all_logs_with_iteration



from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


def simulation_with_control(event_list, train_capacity_dict,pax_path_dict, passenger_objects, all_platforms, all_trains, all_logs, iteration, control_factor_dict, grouped_passengers):

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
            all_platforms, passenger_objects = add_new_passengers_to_platform(event, all_platforms, grouped_passengers,
                                                                              pax_path_dict, passenger_objects)
            control_factor = control_factor_dict.get((event.train_id, event.platform_id), None)
            num_ob_pax, all_logs, all_trains, all_platforms = onboard_passengers(
                event, all_logs, all_trains, all_platforms, control_board_numbers=None, control_factor=control_factor)

            # all_platforms, passenger_objects = add_new_passengers_to_platform(event, all_platforms, grouped_passengers, pax_path_dict, passenger_objects)
            # control_factor = control_factor_dict.get(
            #     (event.train_id, event.platform_id), {'Control': None, 'Onboard': None})
            # if control_factor['Control'] is None:
            #     num_ob_pax, all_logs, all_trains, all_platforms = onboard_passengers(event, all_logs, all_trains, all_platforms, control_factor['Control'])
            #     control_factor_dict[(event.train_id, event.platform_id)] = {'Control': None, 'Onboard': num_ob_pax}
            # else:
            #     num_ob_pax, all_logs, all_trains, all_platforms = onboard_passengers(event, all_logs, all_trains, all_platforms, control_factor['Control'])
            #     control_factor_dict[(event.train_id, event.platform_id)]['Onboard'] = num_ob_pax

    print('total spent time: {} sec'.format(round(time.time() - s_time)))

    # Save output metrics
    # os.makedirs('data', exist_ok=True)
    # pd.DataFrame(train_load_log).to_csv('data/train_load.csv', index=False)
    # pd.DataFrame(platform_queue_log).to_csv('data/platform_queue.csv', index=False)
    # pd.DataFrame(left_behind_log).to_csv('data/left_behind.csv', index=False)
    # pd.DataFrame(passenger_travel_log).to_csv('data/passenger_travel_times.csv', index=False)
    print("Success")
    return control_factor_dict, passenger_objects, all_platforms, all_trains, all_logs







def project_01(x):
    return np.clip(x, 0.0, 1.0)


def main_calculation(case_name, Max_iteration):


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

    control_events = select_platform_train_id(events)


    event_list = generate_event_list(events)
    print('Finish generating event list...')

    all_platform_and_train = control_events[['train_id', 'platform_id']].drop_duplicates().sort_values(['train_id', 'platform_id'])
    vector_to_platform_map =  dict(
        enumerate(
            zip(
            all_platform_and_train['train_id'],
            all_platform_and_train['platform_id']
            )
        )
    )

    control_vector_ini = np.zeros(len(all_platform_and_train)) + 1 ###
    # def convert_control_to_dict(control_vector, old_control_dict):
    #     # print(old_control_dict)
    #     control_factor_dict = {}
    #     for i in range(len(control_vector)):
    #         train_id, platform_id = vector_to_platform_map[i]
    #         if (train_id, platform_id) in old_control_dict:
    #             if  old_control_dict[(train_id, platform_id)]['Onboard'] is not None:
    #                 current_onboard = old_control_dict[(train_id, platform_id)]['Onboard']
    #                 # print(f'current onboard has numbers, control vector: {control_vector[i]}')
    #                 control_factor_dict[(train_id, platform_id)] = {'Control': int(current_onboard * control_vector[i]), 'Onboard': current_onboard}
    #             else:
    #                 # print('current onboard has no numbers')
    #                 control_factor_dict[(train_id, platform_id)] = {'Control': None, 'Onboard': None}
    #         else:
    #             # print('no train id platform id')
    #             control_factor_dict[(train_id, platform_id)] = {'Control': None, 'Onboard': None}
    #     # a=1
    #     return control_factor_dict

    def convert_control_to_dict(control_vector, old_control_dict):
        # print(old_control_dict)
        control_factor_dict = {}
        for i in range(len(control_vector)):
            train_id, platform_id = vector_to_platform_map[i]
            control_factor_dict[(train_id, platform_id)] = control_vector[i]
        return control_factor_dict


    def w_max(control_vector, old_control_dict, iteration):
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
            'left_behind_log_temp': {}
        }
        grouped_passengers = process_passenger_group_by_origin(passenger_df_path)
        print(f'Avg control factor: {round(np.mean(control_vector),8)}')
        control_factor_dict = convert_control_to_dict(control_vector, old_control_dict)

        # print('new control dict', control_factor_dict)
        control_factor_dict, passenger_objects, all_platforms, all_trains, all_logs = simulation_with_control(
            event_list, train_capacity_dict, pax_path_dict, passenger_objects, all_platforms, all_trains,
            all_logs, iteration, control_factor_dict, grouped_passengers)



        print('Finish simulating event list, start to save logs...')
        save_all_logs_with_iteration(all_logs, iteration, case_name)
        all_LBs = [all_logs['left_behind_log_temp'][pax_id]['left_behind_times'] for pax_id in all_logs['left_behind_log_temp']]
        max_lb_pax_times = max(all_LBs)
        new_obj = max_lb_pax_times

        return max_lb_pax_times, control_factor_dict, new_obj

    class ObjectiveWrapper:
        def __init__(self, w_max):
            self.w_max = w_max
            self.iteration = 0
            self.old_control_dict = {}  # store variable

        def __call__(self, x):
            # Example: compute auxiliary variable
            print(f'Evaluate control: {np.mean(x)}')
            x = project_01(x)
            max_lb_pax_times, old_control_dict, new_obj = self.w_max(x, self.old_control_dict, self.iteration)
            self.old_control_dict = old_control_dict
            self.iteration += 1
            print(f'****current Max LB****: {max_lb_pax_times}')
            print(f'****current obj****: {new_obj}')
            print(f'****current iteration****: {self.iteration}')
            if self.iteration > Max_iteration:
                self.shut_down_process()
            return new_obj #

        def shut_down_process(self):
            print(f'Currrnt iteration: {self.iteration}, Shutting down process...')
            exit()


    objective = ObjectiveWrapper(w_max)

    def bayesian_optimization(dim, n_calls=50):

        # Add names here
        space = [
            Real(0.0, 1.0, name=f"x{i}")
            for i in range(dim)
        ]

        @use_named_args(space)
        def skopt_objective(**params):
            # Convert named parameters back to vector
            x = [params[f"x{i}"] for i in range(dim)]
            return objective(x)

        result = gp_minimize(
            skopt_objective,
            space,
            n_calls=n_calls,
            random_state=42,
        )

        return result.x, result.fun

    bayesian_optimization(dim=len(control_vector_ini), n_calls=120)



def select_platform_train_id(events):
    ### only select congestion stations within a period
    lb_log_original = pd.read_csv('output/reference/left_behind_log_iteration_0.csv')
    all_lb_pax = lb_log_original.loc[lb_log_original['left_behind_times'] > 0].copy()
    all_lb_platform = all_lb_pax[['boarding_platform']].drop_duplicates()
    min_time = np.min(all_lb_pax['arrival_time_at_platform'])
    max_time = np.max(all_lb_pax['arrival_time_at_platform'])
    all_lb_platform['boarded_line_id'] = all_lb_pax['boarding_platform'].apply(lambda x: int(x.split('_')[1]))
    all_lb_platform['boarded_direction_id'] = all_lb_pax['boarding_platform'].apply(lambda x: int(x.split('_')[2]))

    all_platforms = pd.read_csv('data/reference/platforms.csv')

    all_lb_platform = all_lb_platform.merge(all_platforms[['platform_id', 'stop_seq']],left_on=['boarding_platform'], right_on=['platform_id'])
    all_lb_platform = all_lb_platform.rename(columns={'stop_seq': 'board_stop_seq'}).drop(columns=['platform_id'])
    all_used_platform = all_platforms.merge(all_lb_platform, how='cross')
    all_used_platform = all_used_platform.loc[
        (all_used_platform['line_id']==all_used_platform['boarded_line_id']) &
        (all_used_platform['direction_id']==all_used_platform['boarded_direction_id']) &
        (all_used_platform['stop_seq']<=all_used_platform['board_stop_seq'])
    ]
    all_used_platform = all_used_platform[['platform_id']].drop_duplicates()
    used_events = events.loc[(events['event_timestamp']<=max_time) & (events['event_timestamp']>=min_time + 20*60)] # plus 20
    used_events = used_events.merge(all_used_platform, on=['platform_id'])
    return used_events


if __name__ == '__main__':
    case_name = 'BYO'

    SIMULATION_START_TIMESTAMP = 6 * 3600
    SIMULATION_END_TIMESTAMP = 9 * 3600

    Max_iteration = 100
    main_calculation(case_name, Max_iteration)
