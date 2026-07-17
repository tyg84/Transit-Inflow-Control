import pandas as pd
import numpy as np
import itertools
import _constant

# Define time-of-day multipliers
def time_multiplier(t):
    if 5*3600 <= t < 7*3600:  # 05:00–07:00
        return 0.5
    elif 7*3600 <= t < 8*3600:  # 07:00–09:00 (Morning Peak)
        return 1.0
    elif 8*3600 <= t < 8.5*3600:  # 07:00–09:00 (Morning Peak)
        return 2
    elif 8.5*3600 <= t < 9*3600:  # 07:00–09:00 (Morning Peak)
        return 3
    elif 32400 <= t < 57600:  # 09:00–16:00
        return 1.0
    elif 57600 <= t < 64800:  # 16:00–18:00 (Evening Peak)
        return 2.5
    elif 64800 <= t < 79200:  # 18:00–22:00
        return 0.7
    else:  # 22:00–05:00
        return 0.2


def generate_demand_data(platforms, case_name, demand_factor, random_seed):
    # -----------------------------
    # OD generation
    # -----------------------------
    TIME_BIN = 900  # 15 min
    DAY_END = 24 * 3600
    ALPHA = _constant.DEMAND_RATE * demand_factor  # global scaling factor

    records = []

    # Load station list
    stations = pd.read_csv('data/manual_input_data/testSubwayStation.csv')
    stations = stations.rename(columns={'station id':'station_id'})
    for t in range(0, DAY_END, TIME_BIN):
        np.random.seed(random_seed+t)
        sudden_peak_prob = 0.05
        mult = time_multiplier(t) * (np.random.choice([1, 2], p=[1-sudden_peak_prob, sudden_peak_prob])) # add randomness to demand, mimic sudden flow in

        # Morning + daytime: residential → work
        if t < 57600:
            for o, d in itertools.permutations(stations.itertuples(), 2):
                if (o.Line != d.Line): # (o.poprating <= 2) or (d.workrating <= 2) or
                    continue
                # if (d.workrating >= 5):
                #     mult = time_multiplier(t) * (np.random.choice([1, 20], p=[1-sudden_peak_prob, sudden_peak_prob])) # add randomness to demand, mimic sudden flow in
                base = o.poprating * d.workrating
                num = int(ALPHA * base * mult)
                if num > 0:
                    records.append([
                        o.station_id,
                        d.station_id,
                        t,
                        t + TIME_BIN,
                        num
                    ])

        # Evening: work → residential
        else:
            for o, d in itertools.permutations(stations.itertuples(), 2):
                base = o.poprating * d.workrating
                if (o.Line != d.Line): # (o.poprating <= 2) or (d.workrating <= 2) or
                    continue
                num = int(ALPHA * base * mult)
                if num > 0:
                    records.append([
                        o.station_id,
                        d.station_id,
                        t,
                        t + TIME_BIN,
                        num
                    ])

    # -----------------------------
    # Final OD table
    # -----------------------------
    od_df = pd.DataFrame(
        records,
        columns=[
            "origin_station_id",
            "destination_station_id",
            "tap_in_time_start",
            "tap_in_time_end",
            "num_passengers"
        ]
    )


    od_df.to_csv(f"data/{case_name}/demands.csv", index=False)


def generate_individual_tap_in_time(demands, case_name, random_seed):

    records = []
    pid = 1

    for _, row in demands.iterrows():
        origin = row["origin_station_id"]
        dest = row["destination_station_id"]
        start = row["tap_in_time_start"]
        end = row["tap_in_time_end"]
        num = row["num_passengers"]
        # assign timestamps uniformly within the interval
        random_seed += 1
        np.random.seed(random_seed)
        timestamps = np.random.randint(start, end, size=num)
        for t in timestamps:
            records.append([pid, origin, dest, t])
            pid += 1

    passenger_df = pd.DataFrame(records, columns=[
        "passenger_id", "origin_station_id", "destination_station_id", "tap_in_timestamp"
    ])


    passenger_df.to_csv(f"data/{case_name}/individual_demands.csv", index=False)
    print("✅ passenger_trips.csv generated.")
    print(f"total num passengers {len(passenger_df)}.")

if __name__ == '__main__':
    case_name = 'reference'
    demand_factor = 1.0
    #############
    platforms = pd.read_csv(f'data/{case_name}/platforms.csv')
    generate_demand_data(platforms, case_name,demand_factor, random_seed = 141)

    ##############
    demands = pd.read_csv(f'data/{case_name}/demands.csv')
    generate_individual_tap_in_time(demands, case_name, random_seed = 142)
