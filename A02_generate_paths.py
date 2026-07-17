"""
import pandas as pd
import networkx as nx

# Sample data
station_line_transfer_time = pd.read_csv("station_line_transfer_times.csv")
station_line_travel_time = pd.read_csv("station_line_travel_times.csv")


# Example input edges (bidirectional)
edges = pd.concat([station_line_transfer_time, station_line_travel_time], ignore_index=True)
# Example direction file
direction = pd.read_csv("platform_travel_times.csv")

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
paths_df.to_csv('paths.csv', index=False)

print(paths_df.head())
"""

import pandas as pd
import networkx as nx

def load_edges(case_name):
    transfer_df = pd.read_csv(f"data/{case_name}/station_line_transfer_times.csv", dtype=str)
    if 'travel_time' in transfer_df.columns:
        transfer_df['travel_time'] = transfer_df['travel_time'].astype(float)
    else:
        transfer_df['travel_time'] = 2.0  # fallback

    travel_df = pd.read_csv(f"data/{case_name}/station_line_travel_times.csv", dtype=str)
    travel_df['travel_time'] = travel_df['travel_time'].astype(float)

    platform_dir_df = pd.read_csv(f"data/{case_name}/platform_travel_times.csv", dtype=str)

    return transfer_df, travel_df, platform_dir_df

def build_graph(transfer_df, travel_df):
    G = nx.Graph()

    for _, row in travel_df.iterrows():
        a = row['from_station_line_id']
        b = row['to_station_line_id']
        w = float(row['travel_time'])
        if a == b:
            continue
        if G.has_edge(a, b):
            if w < G[a][b]['weight']:
                G[a][b]['weight'] = w
        else:
            G.add_edge(a, b, weight=w)

    # add transfer edges (same station, different lines)
    for _, row in transfer_df.iterrows():
        a = row['from_station_line_id']
        b = row['to_station_line_id']
        w = float(row['travel_time'])
        if a == b:
            continue
        if G.has_edge(a, b):
            if w < G[a][b]['weight']:
                G[a][b]['weight'] = w
        else:
            G.add_edge(a, b, weight=w)

    return G

def build_platform_direction_map(platform_dir_df):
    mapping = {}
    mapping_seq = {}
    for _, r in platform_dir_df.iterrows():
        fplat_full = str(r['from_platform_id']).strip()
        tplat_full = str(r['to_platform_id']).strip()
        if '_' not in fplat_full or '_' not in tplat_full:
            continue
        f_station_line = '_'.join(fplat_full.split('_')[:2])
        f_dir = fplat_full.split('_')[2]
        # f_seq = fplat_full.split('_')[3]
        t_station_line = '_'.join(tplat_full.split('_')[:2])
        t_dir = tplat_full.split('_')[2]
        # t_seq = tplat_full.split('_')[3]


        try:
            dir_int = int(f_dir)
        except:
            dir_int = 0

        mapping[(f_station_line, t_station_line)] = dir_int
        # if f_seq not in mapping_seq:
        #     mapping_seq[f_station_line] = int(f_seq)
        # if t_seq not in mapping_seq:
        #     mapping_seq[t_station_line] = int(t_seq)
    return mapping, mapping_seq

def segment_direction_code(f, t, platform_dir_map):
    #0, 1: out/inbound, 2: transfer
    f_station = f.split('_', 1)[0]
    t_station = t.split('_', 1)[0]

    if f_station == t_station and f != t:
        return 2

    if (f, t) in platform_dir_map:
        return int(platform_dir_map[(f, t)])
    if (t, f) in platform_dir_map:
        return 1 - int(platform_dir_map[(t, f)])

    return 0

def generate_all_path_segments(case_name):
    transfer_df, travel_df, platform_dir_df = load_edges(case_name)
    G = build_graph(transfer_df, travel_df)
    platform_dir_map, seq_map = build_platform_direction_map(platform_dir_df)

    nodes = list(G.nodes())
    if not nodes:
        print("error graph has no nodes")
        return

    out_rows = []
    for i, src in enumerate(nodes):
        lengths, paths = nx.single_source_dijkstra(G, src, weight='weight')
        for dst, path in paths.items():
            path_id = 1
            if dst == src:
                continue
            cum_time = 0.0
            for j in range(len(path) - 1):
                f = path[j]
                t = path[j+1]
                line_id_from = int(f.split('_')[1])
                seg_w = float(G[f][t]['weight'])
                cum_time += seg_w
                dir_code = segment_direction_code(f, t, platform_dir_map)
                if dir_code == 2:
                    # transfer station
                    if j > 0 and j < len(path)-2:
                        from_direction_id = segment_direction_code(path[j-1], path[j], platform_dir_map)
                        to_direction_id = segment_direction_code(path[j+1], path[j+2], platform_dir_map)
                    elif j == 0 and j < len(path)-2:
                        # first link is transfer link
                        to_direction_id = segment_direction_code(path[j+1], path[j+2], platform_dir_map)
                        from_direction_id = to_direction_id
                    elif j == len(path)-2:
                        # last link is transfer link
                        from_direction_id = segment_direction_code(path[j-1], path[j], platform_dir_map)
                        to_direction_id = from_direction_id
                    else:
                        raise Exception("Error in defining direction id")

                else:
                    from_direction_id = int(dir_code)
                    to_direction_id = int(dir_code)

                # from_station_seq = seq_map[f]
                # to_station_seq = seq_map[t]
                out_rows.append({
                    'from_station': f,
                    'to_station': t,
                    'line_id': line_id_from,
                    'path_id': path_id,
                    'cumulated_travel_time': cum_time,
                    'from_direction_id': from_direction_id,
                    'to_direction_id': to_direction_id,
                    'if_transfer': 1 if dir_code == 2 else 0,
                    'origin': src,
                    'destination': dst,
                    # 'from_station_seq': from_station_seq,
                    # 'to_station_seq': to_station_seq,
                })

    out_cols = ['origin','destination','path_id','line_id','from_direction_id', 'to_direction_id', 'if_transfer','from_station','to_station','cumulated_travel_time']
    out_df = pd.DataFrame(out_rows, columns=out_cols)



    out_df.to_csv(f'data/{case_name}/paths.csv', index=False)


if __name__ == "__main__":
    case_name = 'reference'

    generate_all_path_segments(case_name)
    print("success")
