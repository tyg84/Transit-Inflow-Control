import pandas as pd
import numpy as np
from itertools import product


# Define time-of-day multipliers
def time_multiplier(t):
    if 18000 <= t < 25200:  # 05:00–07:00
        return 0.5
    elif 25200 <= t < 32400:  # 07:00–09:00 (Morning Peak)
        return 3.0
    elif 32400 <= t < 57600:  # 09:00–16:00
        return 1.0
    elif 57600 <= t < 64800:  # 16:00–18:00 (Evening Peak)
        return 2.5
    elif 64800 <= t < 79200:  # 18:00–22:00
        return 0.7
    else:  # 22:00–05:00
        return 0.2

def generate_demand_data(platforms):
    # Load station list
    stations = sorted(list(set(platforms['station_id'])))

    # 15-minute intervals (96 per day)
    interval_seconds = 15 * 60
    intervals = [(t, t + interval_seconds) for t in range(0, 86400, interval_seconds)]
    base_rate = 2  # average OD passengers per 15 min outside peaks — change as needed
    rows = []
    for origin, dest in product(stations, stations):
        if origin == dest:
            continue

        for start, end in intervals:
            multiplier = time_multiplier(start)
            num_passengers = np.random.poisson(base_rate * multiplier)

            if num_passengers > 0:
                rows.append([origin, dest, start, end, num_passengers])

    df = pd.DataFrame(rows, columns=[
        "origin_station_id", "destination_station_id",
        "tap_in_time_start", "tap_in_time_end", "num_passengers"
    ])

    df.to_csv("data/demands.csv", index=False)


def generate_individual_tap_in_time(demands):

    records = []
    pid = 1

    for _, row in demands.iterrows():
        origin = row["origin_station_id"]
        dest = row["destination_station_id"]
        start = row["tap_in_time_start"]
        end = row["tap_in_time_end"]
        num = row["num_passengers"]
        # assign timestamps uniformly within the interval
        timestamps = np.random.randint(start, end, size=num)
        for t in timestamps:
            records.append([pid, origin, dest, t])
            pid += 1

    passenger_df = pd.DataFrame(records, columns=[
        "passenger_id", "origin_station_id", "destination_station_id", "tap_in_timestamp"
    ])

    passenger_df.to_csv("data/individual_demands.csv", index=False)
    print("✅ passenger_trips.csv generated.")

if __name__ == '__main__':
    #############
    platforms = pd.read_csv('data/platforms.csv')
    generate_demand_data(platforms)

    ##############
    demands = pd.read_csv('data/demands.csv')
    generate_individual_tap_in_time(demands)