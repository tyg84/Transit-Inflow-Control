import pandas as pd


def generate_platforms(raw_data):
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
            platforms_df_list.append(temp_platform)

    platforms_df = pd.concat(platforms_df_list)
    platforms_df = platforms_df.rename(columns={'Line':'line_id',
                                                'station id':'station_id',
                                                'Station name':'station_name',
                                                'Lat':'lat',
                                                'Long':'lon',
                                                'direction':'direction_id'})
    platforms_df['platform_id'] = platforms_df['station_id'].astype('int').astype('str') + '_' + platforms_df['line_id'].astype('int').astype('str') + '_' + platforms_df['direction_id'].astype('int').astype('str')
    platforms_df['station_line_id'] = platforms_df['station_id'].astype('int').astype('str') + '_' + platforms_df[
        'line_id'].astype('int').astype('str')
    column_seq = ['platform_id','station_line_id', 'station_id','line_id','direction_id','station_name','lat','lon']
    platforms_df.to_csv('data/platforms.csv', columns=column_seq, index=False)



def generate_station_pair_travel_time(raw_data):
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
            line_df['from_platform_id'] = line_df['station id'].astype('int').astype('str') + '_' + str(int(line)) + '_' + str(int(direction))
            line_df['to_platform_id'] = line_df['new_station'].astype('int').astype('str') + '_' + str(int(line)) + '_'+ str(int(direction))
            line_df['from_station_line_id'] = line_df['station id'].astype('int').astype('str') + '_' + str(int(line))
            line_df['to_station_line_id'] = line_df['new_station'].astype('int').astype('str') + '_' + str(int(line))
            travel_times_df_list.append(line_df)

    travel_times_df = pd.concat(travel_times_df_list)
    column_seq = ['from_platform_id','to_platform_id','travel_time']
    travel_times_df.to_csv('data/platform_travel_times.csv', columns=column_seq, index=False)


    column_seq = ['from_station_line_id','to_station_line_id','travel_time']
    travel_times_station_line = travel_times_df[column_seq].drop_duplicates()
    travel_times_station_line.to_csv('data/station_line_travel_times.csv', columns=column_seq, index=False)

def construct_transfer_time(platforms):
    platforms['num_lines'] = platforms.groupby('station_id')['line_id'].transform('nunique')
    transfer_station_lines = platforms.loc[platforms['num_lines']>=2, ['station_line_id','station_id','line_id']].copy().sort_values(['station_id']).drop_duplicates()
    transfer_station_lines['to_station_id'] = transfer_station_lines['station_id'].shift(-1)
    transfer_station_lines['to_line_id'] = transfer_station_lines['line_id'].shift(-1)
    transfer_station_lines['to_station_line_id'] = transfer_station_lines['station_line_id'].shift(-1)
    transfer_station_lines = transfer_station_lines.loc[transfer_station_lines['to_station_id'] == transfer_station_lines['station_id']]
    transfer_station_lines['travel_time'] = 2
    transfer_station_lines['from_station_line_id'] = transfer_station_lines['station_line_id']
    column_seq = ['from_station_line_id', 'to_station_line_id', 'travel_time']
    transfer_station_lines.to_csv('data/station_line_transfer_times.csv', columns=column_seq, index=False)




if __name__ == "__main__":
    raw_data = pd.read_csv('data/testSubwayStation.csv')
    generate_platforms(raw_data)
    generate_station_pair_travel_time(raw_data)
    platforms = pd.read_csv('data/platforms.csv')
    construct_transfer_time(platforms)

print("Success")