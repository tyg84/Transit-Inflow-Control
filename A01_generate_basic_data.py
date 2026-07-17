import pandas as pd
import numpy as np
import os
import _constant



def generate_platforms(raw_data, case_name):
    all_lines = set(raw_data['Line'])
    platforms_df_list = []
    all_dir = [0, 1]
    for line in all_lines:
        for direction in all_dir:
            line_df = raw_data.loc[raw_data['Line']==line].copy()
            ascending = True if direction == 0 else False
            line_df = line_df.sort_values(by='station time', ascending=ascending)
            temp_platform = line_df[['Line', 'station id', 'Station name', 'Lat', 'Long']].copy()
            temp_platform['direction'] = direction
            temp_platform['stop_seq'] = np.arange(1, len(temp_platform)+1)
            platforms_df_list.append(temp_platform)

    platforms_df = pd.concat(platforms_df_list)
    platforms_df = platforms_df.rename(columns={'Line':'line_id',
                                                'station id':'station_id',
                                                'Station name':'station_name',
                                                'Lat':'lat',
                                                'Long':'lon',
                                                'direction':'direction_id'})
    platforms_df['platform_id'] = (platforms_df['station_id'].astype('int').astype('str') +
                                   '_' + platforms_df['line_id'].astype('int').astype('str') + '_' +
                                   platforms_df['direction_id'].astype('int').astype('str'))
                                   #+ '_' + platforms_df['stop_seq'].astype('int').astype('str'))
    platforms_df['station_line_id'] = platforms_df['station_id'].astype('int').astype('str') + '_' + platforms_df[
        'line_id'].astype('int').astype('str')
    platforms_df = platforms_df.sort_values(['line_id','direction_id','stop_seq'])
    column_seq = ['platform_id','station_line_id', 'station_id','line_id','direction_id','station_name','lat','lon','stop_seq']


    os.makedirs(f'data/{case_name}', exist_ok=True)

    platforms_df.to_csv(f'data/{case_name}/platforms.csv', columns=column_seq, index=False)



def generate_station_pair_travel_time(raw_data, case_name):
    all_lines = set(raw_data['Line'])
    all_dir = [0, 1]
    travel_times_df_list = []
    for line in all_lines:
        for direction in all_dir:
            line_df = raw_data.loc[raw_data['Line']==line].copy()
            ascending = True if direction == 0 else False
            direction_sign = 1 if direction == 0 else -1
            line_df = line_df.sort_values(by='station time', ascending=ascending)
            line_df['station_time_shift'] = line_df['station time'].shift(-1)
            line_df['new_station'] = line_df['station id'].shift(-1)
            line_df = line_df.dropna()
            line_df['travel_time'] = (line_df['station_time_shift'] - line_df['station time']) * direction_sign

            line_df['from_station_line_id'] = line_df['station id'].astype('int').astype('str') + '_' + str(int(line))
            line_df['to_station_line_id'] = line_df['new_station'].astype('int').astype('str') + '_' + str(int(line))
            line_df['from_platform_seq'] = np.arange(1, len(line_df)+1)
            line_df['to_platform_seq'] = np.arange(2, len(line_df) + 2)
            line_df['from_platform_id'] = line_df['station id'].astype('int').astype('str') + '_' + str(int(line)) + '_' + str(int(direction)) #+'_' + line_df['from_platform_seq'].astype('int').astype('str')
            line_df['to_platform_id'] = line_df['new_station'].astype('int').astype('str') + '_' + str(int(line)) + '_'+ str(int(direction)) #+'_'+ line_df['to_platform_seq'].astype('int').astype('str')
            travel_times_df_list.append(line_df)

    travel_times_df = pd.concat(travel_times_df_list)
    column_seq = ['from_platform_id','to_platform_id','from_platform_seq','to_platform_seq','travel_time']
    travel_times_df.to_csv(f'data/{case_name}/platform_travel_times.csv', columns=column_seq, index=False)


    column_seq = ['from_station_line_id','to_station_line_id','travel_time']
    travel_times_station_line = travel_times_df[column_seq].drop_duplicates()
    travel_times_station_line.to_csv(f'data/{case_name}/station_line_travel_times.csv', columns=column_seq, index=False)

def construct_transfer_time(platforms, case_name):
    platforms['num_lines'] = platforms.groupby('station_id')['line_id'].transform('nunique')
    transfer_station_lines = platforms.loc[platforms['num_lines']>=2, ['station_line_id','station_id','line_id']].copy().sort_values(['station_id']).drop_duplicates()
    transfer_station_lines['to_station_id'] = transfer_station_lines['station_id'].shift(-1)
    transfer_station_lines['to_line_id'] = transfer_station_lines['line_id'].shift(-1)
    transfer_station_lines['to_station_line_id'] = transfer_station_lines['station_line_id'].shift(-1)
    transfer_station_lines = transfer_station_lines.loc[transfer_station_lines['to_station_id'] == transfer_station_lines['station_id']]
    transfer_station_lines['travel_time'] = 2
    transfer_station_lines['from_station_line_id'] = transfer_station_lines['station_line_id']
    column_seq = ['from_station_line_id', 'to_station_line_id', 'travel_time']
    transfer_station_lines.to_csv(f'data/{case_name}/station_line_transfer_times.csv', columns=column_seq, index=False)




if __name__ == "__main__":

    case_name = 'reference'

    ##################
    raw_data = pd.read_csv(_constant.manual_input_path('testSubwayStation.csv'))
    generate_platforms(raw_data, case_name)
    generate_station_pair_travel_time(raw_data, case_name)
    #################
    platforms = pd.read_csv(f'data/{case_name}/platforms.csv')
    construct_transfer_time(platforms, case_name)
    #################
