import pandas as pd
import networkx as nx

# Sample data
station_line_transfer_time = pd.read_csv("data/station_line_transfer_times.csv")
station_line_travel_time = pd.read_csv("data/station_line_travel_times.csv")


# Example input edges (bidirectional)
edges = pd.concat([station_line_transfer_time, station_line_travel_time], ignore_index=True)
# Example direction file
direction = pd.read_csv("data/platform_travel_times.csv")

# Step 1: Build undirected graph from edges
G = nx.Graph()
for _, row in edges.iterrows():
    G.add_edge(row['from_station_line_id'], row['to_station_line_id'], weight=row['travel_time'])

# Step 2: Compute all shortest paths (between different stations)
results = []
stations = list(G.nodes())

for i, src in enumerate(stations):
    src_station = src.split('_')[0]
    for dst in stations[i+1:]:
        dst_station = dst.split('_')[0]
        if src_station == dst_station:
            continue  # skip same station id

        try:
            length = nx.shortest_path_length(G, src, dst, weight='weight')
            results.append({
                'from_station_line_id': src,
                'to_station_line_id': dst,
                'cumulated_travel_time': length
            })
        except nx.NetworkXNoPath:
            continue  # skip disconnected nodes

paths_df = pd.DataFrame(results)

# Step 3: Add direction column by checking if path exists in direction df
# Build a quick lookup
directed_edges = set(zip(direction['from_platform_id'].str[:-2], direction['to_platform_id'].str[:-2]))

def get_direction(row):
    f, t = row['from_station_line_id'], row['to_station_line_id']
    if (f, t) in directed_edges:
        return 'forward'
    elif (t, f) in directed_edges:
        return 'backward'
    else:
        return 'unknown'

paths_df['direction'] = paths_df.apply(get_direction, axis=1)

# Step 4: Save to CSV
paths_df.to_csv('data/paths.csv', index=False)

print(paths_df.head())
