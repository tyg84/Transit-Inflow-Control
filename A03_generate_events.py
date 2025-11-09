import pandas as pd
import _constant

def generate_events(headway, platforms, platform_travel_times):
    headway['start_timestamp'] = pd.to_timedelta(headway['start_time']).dt.total_seconds().astype(int)
    headway['end_timestamp'] = pd.to_timedelta(headway['end_time']).dt.total_seconds().astype(int)
    headway_group = headway.groupby(['line_id','direction_id'])
    first_stop_events = {'line_id':[],'direction_id':[],'train_id':[],'timestamp':[]}
    for idx, group in headway_group:
        group = group.sort_values('start_timestamp', ascending=True)
        interval_id = 0
        end_timestamp = None
        last_train_id = 0
        interval_old = 0
        for group_idx, row in group.iterrows():
            interval = row['headway']
            if interval_id == 0:
                start_timestamp = row['start_timestamp']
            else:
                start_timestamp = end_timestamp + interval_old + interval
                last_train_id = first_stop_events['train_id'][-1]
            end_timestamp = row['end_timestamp']
            departure_timestamps = list(range(start_timestamp, end_timestamp + interval, interval))
            first_stop_events['line_id'] += [row['line_id']] * len(departure_timestamps)
            first_stop_events['direction_id'] += [row['direction_id']] * len(departure_timestamps)
            first_stop_events['train_id'] += list(range(last_train_id+1, len(departure_timestamps)+last_train_id+1))
            first_stop_events['timestamp'] += departure_timestamps
            interval_id += 1
            interval_old = interval

    first_stop_events_df = pd.DataFrame(first_stop_events)
    platform_travel_times['line_id'] = platform_travel_times['from_platform_id'].str.split('_').str[1].astype(int)
    platform_travel_times['station_id'] = platform_travel_times['from_platform_id'].str.split('_').str[0].astype(int)
    platform_travel_times['direction_id'] = platform_travel_times['from_platform_id'].str.split('_').str[2].astype(int)
    other_stop_events = first_stop_events_df.merge(platform_travel_times, on = ['line_id','direction_id'])
    other_stop_events = other_stop_events.sort_values(['line_id','direction_id','train_id','station_id'])
    other_stop_events['travel_time_sec'] = other_stop_events['travel_time'] * 60
    other_stop_events['arrival_timestamp'] = other_stop_events.groupby(['line_id','direction_id','train_id'])['travel_time_sec'].cumsum() + _constant.DEFAULT_PLATFORM_STOP_TIME
    arrival_events = other_stop_events[['train_id','to_platform_id','arrival_timestamp']].rename(columns={
        'to_platform_id': 'platform_id',
        'arrival_timestamp':'event_timestamp'}).copy()
    first_arrival_event = first_stop_events_df.copy().rename(columns={'timestamp': 'train_dispatch_time'})
    first_platforms = platforms.loc[platforms['stop_seq']==1,['line_id','direction_id','platform_id']].copy()
    first_arrival_event = first_arrival_event.merge(first_platforms, on=['line_id','direction_id'])
    arrival_events = pd.concat([first_arrival_event[['train_id','platform_id']], arrival_events])
    arrival_events['line_id'] = arrival_events['platform_id'].str.split('_').str[1].astype(int)
    arrival_events['station_id'] = arrival_events['platform_id'].str.split('_').str[0].astype(int)
    arrival_events['direction_id'] = arrival_events['platform_id'].str.split('_').str[2].astype(int)
    arrival_events = arrival_events.merge(first_arrival_event[['line_id','direction_id','train_id','train_dispatch_time']].drop_duplicates(),on=['line_id','direction_id','train_id'])

    arrival_events.loc[~arrival_events['event_timestamp'].isna(), 'event_timestamp'] += arrival_events.loc[~arrival_events['event_timestamp'].isna(), 'train_dispatch_time']
    arrival_events.loc[arrival_events['event_timestamp'].isna(), 'event_timestamp'] = arrival_events.loc[arrival_events['event_timestamp'].isna(), 'train_dispatch_time']
    arrival_events = arrival_events.sort_values(['line_id','direction_id','train_id','event_timestamp'])

    ############ departure events
    departure_events = arrival_events.copy()
    departure_events['event_timestamp'] += _constant.DEFAULT_PLATFORM_STOP_TIME
    arrival_events['event_type'] = 'Arrival'
    departure_events['event_type'] = 'Departure'
    all_events = pd.concat([arrival_events, departure_events])
    int_col = ['line_id','station_id','direction_id','event_timestamp']
    for col in int_col:
        all_events[col] = all_events[col].astype(int)
    output_col = ['line_id', 'direction_id', 'train_id','station_id','platform_id','event_timestamp','event_type']
    all_events = all_events.sort_values(output_col)
    all_events['train_id'] = all_events['line_id'].astype('str') + '_' + all_events['direction_id'].astype('str') + '_' + all_events['train_id'].astype('str')
    all_events.to_csv('data/events.csv', columns=output_col, index=False)


if __name__ == '__main__':
    platforms = pd.read_csv('data/platforms.csv')
    headway = pd.read_csv('data/headway.csv')
    platform_travel_times = pd.read_csv('data/platform_travel_times.csv')
    generate_events(headway, platforms, platform_travel_times)