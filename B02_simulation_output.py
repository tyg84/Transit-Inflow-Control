import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('data/operator_summary', exist_ok=True)

train_load = pd.read_csv('data/train_load.csv')
platform_queue = pd.read_csv('data/platform_queue.csv')
left_behind = pd.read_csv('data/left_behind.csv')
passenger_travel = pd.read_csv('data/passenger_travel_times.csv')

train_load.groupby('train_id').agg(
    max_load=('train_load', 'max'),
    avg_load=('train_load', 'mean')
).reset_index().to_csv('data/operator_summary/train_load_summary.csv', index=False)

for train_id, df in train_load.groupby('train_id'):
    plt.figure()
    plt.plot(df['timestamp']/3600, df['train_load'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Train Load')
    plt.title(f'Train Load Over Time: {train_id}')
    plt.grid(True)
    plt.savefig(f'data/operator_summary/train_load_{train_id}.png')
    plt.close()

platform_queue.groupby('platform_id').agg(
    max_queue=('queue_length', 'max'),
    avg_queue=('queue_length', 'mean')
).reset_index().to_csv('data/operator_summary/platform_queue_summary.csv', index=False)

for platform_id, df in platform_queue.groupby('platform_id'):
    plt.figure()
    plt.plot(df['timestamp']/3600, df['queue_length'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Queue Length')
    plt.title(f'Platform Queue Over Time: {platform_id}')
    plt.grid(True)
    plt.savefig(f'data/operator_summary/platform_queue_{platform_id}.png')
    plt.close()

left_behind.groupby('platform_id').agg(
    total_left=('passenger_id', 'count')
).reset_index().to_csv('data/operator_summary/left_behind_summary.csv', index=False)


complete_travel = passenger_travel.dropna(subset=['boarding_time', 'alighting_time'])
complete_travel['total_travel_time'] = complete_travel['total_travel_time']  # already exists

complete_travel.groupby('origin_station_id').agg(
    avg_travel_time=('total_travel_time', 'mean'),
    min_travel_time=('total_travel_time', 'min'),
    max_travel_time=('total_travel_time', 'max')
).reset_index().to_csv('data/operator_summary/passenger_travel_summary.csv', index=False)

plt.figure()
plt.hist(complete_travel['total_travel_time']/60, bins=50)
plt.xlabel('Travel Time (minutes)')
plt.ylabel('Number of Passengers')
plt.title('Passenger Travel Time Distribution')
plt.grid(True)
plt.savefig('data/operator_summary/passenger_travel_time_distribution.png')
plt.close()

print("Success")
