
"""
New subway simulation: "train-based" simulation using only the five CSV files produced
earlier:

Required files (in same folder):
 - platforms.csv
 - platform_travel_times.csv
 - station_line_travel_times.csv
 - station_line_transfer_times.csv
 - all_paths_segments.csv   (or paths.csv -- this script will try both)

What this script does:
 - Builds a graph of station_line nodes (node id example: "103_1")
 - Reconstructs ordered station lists for each line (both directions) so trains can traverse
 - Spawns passengers at stations over time (uses optional station attributes if available)
 - Runs scheduled trains on each line & direction; trains pick up passengers from the proper platform
 - Handles transfers with a transfer-time delay (uses station_line_transfer_times.csv)
 - Outputs aggregate stats at the end: arrivals, transfers, per-station totals and top stations

Notes / Assumptions:
 - If you have an external station attributes file named "station_attributes.csv" with columns:
     station_id,poprating,workrating
   it will be loaded and used to control spawning rates and destination weighting. If not found,
   the script will derive simple default rates from platform counts (so it will still run).
 - Frequencies (minutes between trains) and capacities per line must be supplied by the user
   in `LINE_FREQ` and `LINE_CAPACITY` dictionaries below. If you don't know them, sensible
   defaults are provided for up to 10 lines.
 - Time is simulated in integer minutes for the interval START_MIN .. END_MIN (inclusive start,
   exclusive end). The original used 330 to 800; defaults follow that (05:30 - 13:20).
 - Travel times in CSVs are treated as minutes (floats allowed).
 - The script runs deterministically unless you change randomness seeds.

Save this file, ensure the five CSVs are present, then run with Python 3.9+.
"""

import csv
import math
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import pandas as pd
import networkx as nx

random.seed(1)

# ---------------------------
# USER PARAMETERS (edit if desired)
# ---------------------------
START_MIN = 330       # simulation start minute (05:30)
END_MIN = 800         # simulation end minute (13:20)
# default line frequencies (minutes between trains). Override with your values if needed.
LINE_FREQ = {
    # line_id : minutes_between_trains
    1: 2.5,   # example; set according to your network
    2: 2.5,
    3: 5.0,
    4: 1.8
}
# default line capacities (passengers per train)
LINE_CAPACITY = {
    1: 2480,
    2: 2480,
    3: 1860,
    4: 1860
}
# station population coefficient used in spawn curve if station_attributes missing
STATION_POPCOEFF = 20

# ---------------------------
# Helper dataclasses
# ---------------------------

@dataclass
class Passenger:
    origin_station: int
    destination_station: int
    path_station_lines: List[str]  # list of station_line nodes e.g. ['103_1', '105_1', '108_1', '108_4', ...]
    next_index: int = 1            # index in path_station_lines that passenger will head to next (0 is origin)

@dataclass
class Train:
    line_id: int
    direction: int  # 0 or 1
    route: List[str]           # ordered station_line nodes for this direction
    position_index: int        # index in route of current station_line node (train is at station when index points)
    capacity: int
    onboard: List[Passenger] = field(default_factory=list)
    next_departure_time: float = 0.0  # if dwell or schedule affects departure

# ---------------------------
# Load CSVs
# ---------------------------

def load_csv_files():
    platforms = pd.read_csv("data/platforms.csv", dtype=str)
    ptt = pd.read_csv("data/platform_travel_times.csv", dtype=str)  # platform to platform with direction suffix
    sltt = pd.read_csv("data/station_line_travel_times.csv", dtype=str)  # station-line to station-line (undirected)
    sl_trans = pd.read_csv("data/station_line_transfer_times.csv", dtype=str)
    # try both filename variants for paths
    try:
        paths = pd.read_csv("data/all_paths_segments.csv", dtype=str)
    except FileNotFoundError:
        try:
            paths = pd.read_csv("data/paths.csv", dtype=str)
        except FileNotFoundError:
            paths = None

    # ensure numeric columns converted
    if 'travel_time' in ptt.columns:
        ptt['travel_time'] = ptt['travel_time'].astype(float)
    if 'travel_time' in sltt.columns:
        sltt['travel_time'] = sltt['travel_time'].astype(float)
    if 'travel_time' in sl_trans.columns:
        sl_trans['travel_time'] = sl_trans['travel_time'].astype(float)

    return platforms, ptt, sltt, sl_trans, paths

platforms_df, platform_travel_df, sl_travel_df, sl_transfer_df, all_paths_df = load_csv_files()

# ---------------------------
# Build graph of station_line nodes
# ---------------------------

G = nx.Graph()
# add edges from station_line_travel_times.csv
for _, r in sl_travel_df.iterrows():
    a = r['from_station_line_id']
    b = r['to_station_line_id']
    w = float(r['travel_time'])
    if a == b:
        continue
    if G.has_edge(a, b):
        if w < G[a][b]['weight']:
            G[a][b]['weight'] = w
    else:
        G.add_edge(a, b, weight=w)

# add transfer edges
for _, r in sl_transfer_df.iterrows():
    a = r['from_station_line_id']
    b = r['to_station_line_id']
    w = float(r['travel_time'])
    if a == b:
        continue
    if G.has_edge(a, b):
        if w < G[a][b]['weight']:
            G[a][b]['weight'] = w
    else:
        G.add_edge(a, b, weight=w)

# list of station_line nodes
station_line_nodes = list(G.nodes())

# ---------------------------
# Utilities for line station ordering (reconstruct route for each line)
# ---------------------------
def station_line_to_tuple(s: str) -> Tuple[int, int]:
    # parse '103_1' -> (103, 1)
    parts = s.split('_')
    return (int(parts[0]), int(parts[1]))

# Build adjacency per line using sl_travel_df rows that share line id
line_adj = defaultdict(lambda: defaultdict(list))  # line_id -> node -> list(neighbours)
for _, r in sl_travel_df.iterrows():
    a = r['from_station_line_id']
    b = r['to_station_line_id']
    # ensure same line
    try:
        _, line_a = station_line_to_tuple(a)
        _, line_b = station_line_to_tuple(b)
    except Exception:
        continue
    if line_a != line_b:
        continue
    line_adj[line_a][a].append(b)
    line_adj[line_a][b].append(a)

# For each line, reconstruct an ordered route by walking from an endpoint
line_routes_forward = {}  # line_id -> ordered list of station_line nodes in "forward" sense
for line_id, adj in line_adj.items():
    # find endpoints (degree 1)
    endpoints = [n for n, nbrs in adj.items() if len(nbrs) == 1]
    if not endpoints:
        # cycle or single station - try arbitrary ordering by BFS
        nodes = list(adj.keys())
        if not nodes:
            continue
        start = nodes[0]
    else:
        start = endpoints[0]

    # walk greedily to build an ordered list
    ordered = []
    visited = set()
    cur = start
    prev = None
    while True:
        ordered.append(cur)
        visited.add(cur)
        nbrs = [n for n in adj[cur] if n != prev]
        if not nbrs:
            break
        prev = cur
        cur = nbrs[0]
        if cur in visited:
            break
    line_routes_forward[line_id] = ordered

# Build reverse route as the other direction
line_routes = {}
for lid, forward in line_routes_forward.items():
    line_routes[lid] = {
        0: forward,           # direction 0 -> forward list
        1: list(reversed(forward))  # direction 1 -> reverse
    }

# ---------------------------
# Platform direction mapping: which station_line -> station_line corresponds to platform forward/backward
# Use platform_travel_df which has from_platform_id and to_platform_id like '103_1_0'
# ---------------------------
platform_dir_map = {}  # (from_station_line, to_station_line) -> dir_int (0 or 1) based on from_platform suffix
for _, r in platform_travel_df.iterrows():
    fplat = str(r['from_platform_id']).strip()
    tplat = str(r['to_platform_id']).strip()
    if '_' not in fplat or '_' not in tplat:
        continue
    f_sl, f_dir = fplat.rsplit('_', 1)
    t_sl, t_dir = tplat.rsplit('_', 1)
    try:
        dir_int = int(f_dir)
    except:
        dir_int = 0
    platform_dir_map[(f_sl, t_sl)] = dir_int
    # Note: we only store the mapping as given (one-way)

def segment_direction_code(f_sl: str, t_sl: str) -> int:
    """Return 2 for transfer (same station id different line),
       else 0 if (f,t) in platform_dir_map,
       else 1 if (t,f) in platform_dir_map,
       else 0 fallback."""
    fs = int(f_sl.split('_', 1)[0])
    ts = int(t_sl.split('_', 1)[0])
    if fs == ts and f_sl != t_sl:
        return 2
    if (f_sl, t_sl) in platform_dir_map:
        return int(platform_dir_map[(f_sl, t_sl)])
    if (t_sl, f_sl) in platform_dir_map:
        return 1 - int(platform_dir_map[(t_sl, f_sl)])
    return 0

# ---------------------------
# Station attributes & spawn functions
# ---------------------------
# Try to load station_attributes.csv for poprating/workrating. If absent, derive defaults.
try:
    station_attrs = pd.read_csv("data/station_attributes.csv", dtype={'station_id': int})
    # expected columns: station_id, poprating, workrating
    station_attrs = station_attrs.set_index('station_id')
    have_attrs = True
except FileNotFoundError:
    # derive station list and defaults from platforms
    have_attrs = False
    station_ids = sorted({int(s.split('_',1)[0]) for s in platforms_df['station_id'].astype(str).unique()})
    # default poprating = number of platforms at station (proxy)
    counts = defaultdict(int)
    for _, r in platforms_df.iterrows():
        counts[int(r['station_id'])] += 1
    rows = []
    for sid in station_ids:
        rows.append({'station_id': sid, 'poprating': counts[sid], 'workrating': 1})
    station_attrs = pd.DataFrame(rows).set_index('station_id')

# Normalize workrating to distribution for destination choice
workratings = station_attrs['workrating'].astype(float)
dest_weights = workratings / workratings.sum()

# curve function roughly matching original shape, returns base passengers multiplier
def curve(t_min: int):
    # same formula as original script, using STATION_POPCOEFF
    return (t_min - 330) ** 2 * (1 - ((t_min - 330) / 240)) / 2400 * STATION_POPCOEFF

# ---------------------------
# Platform queues and transfer buffer
# platform_queues[station_id][line][direction] -> deque of Passenger references
# transfer_buffer: list of tuples (ready_time_min, passenger, target_station_line)
# ---------------------------
platform_queues: Dict[int, Dict[int, Dict[int, deque]]] = defaultdict(lambda: defaultdict(lambda: {0: deque(), 1: deque()}))
transfer_buffer: List[Tuple[int, Passenger, str]] = []  # (ready_time_min, passenger, target_station_line)

# arrivals and transfers counters
station_arrivals = defaultdict(int)
station_transfers = defaultdict(int)

# Precompute shortest path station_line lists for routing using nx.shortest_path on G
# We will compute shortest paths on-demand with caching to avoid huge memory usage.
sp_cache: Dict[Tuple[int, int], List[str]] = {}

def get_shortest_stationline_path(origin_station_id: int, dest_station_id: int) -> Optional[List[str]]:
    """Return the shortest station_line path (list of station_line nodes) from any platform at origin
    to any platform at destination. Chooses the path with minimal travel time."""
    key = (origin_station_id, dest_station_id)
    if key in sp_cache:
        return sp_cache[key]
    # candidate start nodes: any station_line node where station id matches origin
    starts = [n for n in station_line_nodes if int(n.split('_',1)[0]) == origin_station_id]
    ends = [n for n in station_line_nodes if int(n.split('_',1)[0]) == dest_station_id]
    best_path = None
    best_len = float('inf')
    for s in starts:
        for e in ends:
            try:
                length = nx.shortest_path_length(G, s, e, weight='weight')
                if length < best_len:
                    p = nx.shortest_path(G, s, e, weight='weight')
                    best_len = length
                    best_path = p
            except nx.NetworkXNoPath:
                continue
    if best_path is None:
        sp_cache[key] = None
    else:
        sp_cache[key] = best_path
    return sp_cache[key]

# ---------------------------
# Trains initialization
# For each line and direction we will spawn trains according to LINE_FREQ.
# We'll maintain for each running train:
#   - current station index (position in route list)
#   - time until next arrival/departure as we step through travel times
# For simplicity trains start at the terminal stations at times that match frequency so they are spaced evenly.
# ---------------------------

# Build per-line direction train schedules (list of Train objects)
trains: List[Train] = []

# Default scheduler: earliest departure at START_MIN, then every LINE_FREQ minutes.
def initialize_trains(line_routes, LINE_FREQ, LINE_CAPACITY):
    trains_local = []
    for line_id, dirs in line_routes.items():
        # determine frequency and capacity for this line
        freq = LINE_FREQ.get(line_id, 5.0)        # fallback 5 minutes
        cap = LINE_CAPACITY.get(line_id, 1000)    # fallback
        for direction, route in dirs.items():
            if not route:
                continue
            # We'll compute how many trains to instantiate so that they are spaced across time window.
            # Start trains at time = START_MIN, START_MIN + offset where offset in [0,freq)
            # Instead of creating many trains explicitly, we create trains that loop continuously across the simulation time.
            # We'll create one train per desired start offset in [0, freq) quantized to 1 minute increments.
            # Number of distinct trains to simulate per direction = ceil((END_MIN-START_MIN) / freq / 2) minimum 1
            n_trains = max(1, int(math.ceil((END_MIN - START_MIN) / freq / 3)))
            # place trains evenly along the route by offsetting their starting index proportional to n_trains
            L = len(route)
            for tindex in range(n_trains):
                # position index set so trains are spaced along route
                # distribute between 0..L-1
                pos = int((tindex / n_trains) * L) % L
                train = Train(
                    line_id=int(line_id),
                    direction=int(direction),
                    route=route,
                    position_index=pos,
                    capacity=int(cap),
                    onboard=[]
                )
                trains_local.append(train)
    return trains_local

trains = initialize_trains(line_routes, LINE_FREQ, LINE_CAPACITY)

# ---------------------------
# Helper: get single-segment travel time between station_line a->b from the graph
# ---------------------------
def seg_travel_time(a: str, b: str) -> float:
    if G.has_edge(a, b):
        return float(G[a][b]['weight'])
    # fallback to 1 minute if missing
    return 1.0

# ---------------------------
# Spawn passengers each minute
# For each station (physical station id), spawn a number of passengers based on curve(time) * poprating
# Then choose destination stations randomly using dest_weights (workrating based)
# For each passenger, compute shortest station_line path and enqueue them to the appropriate starting platform queue.
# ---------------------------

# Build list of unique station ids and mapping to platform station_line nodes available
unique_station_ids = sorted({int(s.split('_',1)[0]) for s in platforms_df['station_id'].astype(str).unique()})
station_to_stationlines = defaultdict(list)
for n in station_line_nodes:
    sid = int(n.split('_',1)[0])
    station_to_stationlines[sid].append(n)

# For mapping next hop to platform queue direction
def get_boarding_direction(origin_sl: str, next_sl: str) -> int:
    # use platform_dir_map logic
    return segment_direction_code(origin_sl, next_sl)

def enqueue_new_passenger(current_time_min: int, origin_sid: int, dest_sid: int):
    # find shortest station_line path
    path = get_shortest_stationline_path(origin_sid, dest_sid)
    if not path or len(path) < 2:
        return False
    # origin platform is path[0], next target is path[1]
    origin_sl = path[0]
    next_sl = path[1]
    dir_code = get_boarding_direction(origin_sl, next_sl)  # 0 or 1 or 2(transfer)
    # if dir_code == 2 that's odd for initial boarding; treat as need to transfer immediately -> skip
    if dir_code == 2:
        # try to find another platform at origin that leads to next_sl
        # fallback: pick first available platform list and assume direction 0
        dir_code = 0
    passenger = Passenger(
        origin_station=origin_sid,
        destination_station=dest_sid,
        path_station_lines=path,
        next_index=1
    )
    # origin_sl is like '103_1' -> station id part get station integer
    origin_station_id = int(origin_sl.split('_',1)[0])
    origin_line = int(origin_sl.split('_',1)[1])
    # put into platform queue for that station and line and direction
    platform_queues[origin_station_id][origin_line][dir_code].append(passenger)
    return True

# ---------------------------
# Simulation main loop
# We simulate minute-by-minute.
# At each minute:
#  - Activate transfer_buffer: move ready passengers to their target platform_queue
#  - Spawn new passengers at stations according to curve(time) * poprating
#  - Move trains: for each train in trains, if it is at a station this minute, do boarding/alighting.
#    Then advance the train along the route consuming travel time. We'll model travel by computing
#    a per-train 'time_to_next' float which counts down; for simplicity we step per-minute and
#    decrement by 1 each minute; when zero, arrive at next station.
# For simplicity we treat dwell time as 0.5 minute (rounded) before departure.
# ---------------------------

# Maintain per-train time_to_next (minutes until arrival at current station index)
train_time_to_next = {}
train_dwell_remaining = {}  # dwell minutes at station before departure

# Initialize travel times by computing next edge travel time for each train position
for idx, tr in enumerate(trains):
    # determine next index in route depending on direction:
    cur = tr.position_index
    next_idx = (cur + 1) % len(tr.route)
    a = tr.route[cur]
    b = tr.route[next_idx]
    train_time_to_next[idx] = seg_travel_time(a, b)  # float minutes until arrival at next station
    train_dwell_remaining[idx] = 0.0

# Counters for statistics
total_spawned = 0
total_boarded = 0
total_arrivals = 0
total_transfers = 0

# minute loop
for time_min in range(START_MIN, END_MIN):
    # 1) process transfer_buffer
    if transfer_buffer:
        new_buffer = []
        for ready_time, passenger, target_sl in transfer_buffer:
            if ready_time <= time_min:
                # push passenger to platform queue of target_sl
                target_station = int(target_sl.split('_',1)[0])
                target_line = int(target_sl.split('_',1)[1])
                # determine platform direction for next hop (passenger.next_index points to stationline they will head next)
                # but when transferring, their path[next_index] should match target_sl; we compute next hop direction
                if passenger.next_index < len(passenger.path_station_lines):
                    next_sl = passenger.path_station_lines[passenger.next_index]
                    # direction for boarding at target_sl -> next_sl
                    dcode = get_boarding_direction(target_sl, next_sl)
                else:
                    # if no further hop, they are at destination already
                    dcode = 0
                platform_queues[target_station][target_line][dcode].append(passenger)
            else:
                new_buffer.append((ready_time, passenger, target_sl))
        transfer_buffer = new_buffer

    # 2) spawn new passengers at this minute
    base = curve(time_min)
    # iterate through physical stations
    for sid in unique_station_ids:
        poprating = float(station_attrs.loc[sid, 'poprating']) if sid in station_attrs.index else 1.0
        expected = max(0, int(base) * int(max(1, poprating)))
        if expected == 0:
            continue
        total_spawned += expected
        # sample destinations
        dest_indices = random.choices(
            population=list(station_attrs.index),
            weights=dest_weights,
            k=expected
        )
        for d in dest_indices:
            d = int(d)
            if d == sid:
                continue
            # get path; if none, skip
            path = get_shortest_stationline_path(sid, d)
            if not path:
                continue
            # enqueue passenger
            enqueue_new_passenger(time_min, sid, d)

    # 3) train operations: for each train, decrement time_to_next; if arrived at station, handle alight/board/dwell
    for t_idx, tr in enumerate(trains):
        # decrement travel time to next (if > 0)
        ttn = train_time_to_next.get(t_idx, 0.0)
        if ttn > 0:
            train_time_to_next[t_idx] = max(0.0, ttn - 1.0)
            # if not yet arrived, continue to next train
            if train_time_to_next[t_idx] > 0.0:
                continue
        # train has arrived at its next station index (we treat position_index as where the train is now)
        # POSITION: tr.position_index indicates the station the train currently is at.
        # Perform alighting: remove passengers whose next hop corresponds to this station arrival
        cur_idx = tr.position_index
        cur_sl = tr.route[cur_idx]
        cur_station = int(cur_sl.split('_',1)[0])
        # Find passengers to alight: those whose next_index points to this station_line
        remaining_onboard = []
        for p in tr.onboard:
            # p.path_station_lines[p.next_index] should equal cur_sl (the station we arrived at)
            if p.next_index < len(p.path_station_lines) and p.path_station_lines[p.next_index] == cur_sl:
                # They have arrived at an intermediate station on their path.
                # Advance their next_index (they have "arrived" at station_line cur_sl)
                p.next_index += 1
                # If arrived at final destination station (physical station id)
                if int(p.path_station_lines[-1].split('_',1)[0]) == p.destination_station and p.next_index >= len(p.path_station_lines):
                    # passenger completed journey
                    station_arrivals[p.destination_station] += 1
                    total_arrivals += 1
                else:
                    # if next hop is a transfer (i.e., next hop is different line but same station),
                    # we place into transfer_buffer for transfer_time then enqueue to that platform
                    if p.next_index < len(p.path_station_lines):
                        next_sl = p.path_station_lines[p.next_index]
                        # determine if this step is a transfer (same station id)
                        fs = int(cur_sl.split('_',1)[0])
                        ns = int(next_sl.split('_',1)[0])
                        if fs == ns and cur_sl != next_sl:
                            # it's a transfer
                            # find transfer time from sl_transfer_df
                            # find row where from_station_line_id == cur_sl and to_station_line_id == next_sl (or reverse)
                            tt_rows = sl_transfer_df[
                                ((sl_transfer_df['from_station_line_id'] == cur_sl) & (sl_transfer_df['to_station_line_id'] == next_sl)) |
                                ((sl_transfer_df['from_station_line_id'] == next_sl) & (sl_transfer_df['to_station_line_id'] == cur_sl))
                            ]
                            transfer_time = float(tt_rows['travel_time'].iloc[0]) if not tt_rows.empty else 2.0
                            ready_time = time_min + math.ceil(transfer_time)
                            transfer_buffer.append((ready_time, p, next_sl))
                            station_transfers[fs] += 1
                            total_transfers += 1
                        else:
                            # next hop is along a line (passenger will wait on platform of current station for next train)
                            # place into platform queue for current station & correct line/direction
                            # determine the line of next_sl (e.g., '108_4' -> line 4)
                            next_line = int(next_sl.split('_',1)[1])
                            # compute boarding dir for current station (from cur_sl to next_sl)
                            dcode = get_boarding_direction(cur_sl, next_sl)
                            platform_queues[cur_station][next_line][dcode].append(p)
                    else:
                        # shouldn't happen - safe guard
                        station_arrivals[p.destination_station] += 1
                        total_arrivals += 1
                # passenger alighted, do not keep in onboard
            else:
                remaining_onboard.append(p)
        tr.onboard = remaining_onboard

        # Boarding: allow boarding from this station's queues for this train's line and its next movement direction
        # Determine next station for the train (where it will head next) to resolve direction code used by platform_queues
        nxt_idx = (cur_idx + 1) % len(tr.route)
        next_sl_for_train = tr.route[nxt_idx]
        # Determine the line the train is on:
        train_line = tr.line_id
        # boarding direction expected at this station to move to next_sl_for_train
        boarding_dir_code = get_boarding_direction(tr.route[cur_idx], next_sl_for_train)
        # board up to available capacity
        cap_free = tr.capacity - len(tr.onboard)
        boarded = 0
        if cap_free > 0:
            q = platform_queues[cur_station][train_line][boarding_dir_code]
            # board in FIFO order
            while q and boarded < cap_free:
                passenger = q.popleft()
                tr.onboard.append(passenger)
                boarded += 1
                total_boarded += 1
        # set dwell time minimal (0 or 1 minute)
        dwell = 0.5
        train_dwell_remaining[t_idx] = dwell

        # prepare to depart: set time_to_next to travel time to next station
        # if route length is 1 then no movement
        if len(tr.route) > 1:
            travel = seg_travel_time(tr.route[cur_idx], tr.route[nxt_idx])
            train_time_to_next[t_idx] = travel  # float minutes
            # advance position index to next station *when* arrival occurs; for simplicity we advance now
            tr.position_index = nxt_idx
        else:
            train_time_to_next[t_idx] = 1.0

    # End minute loop iteration
# ---------------------------
# Simulation finished; report stats
# ---------------------------

print("Simulation complete.")
print(f"Time window: {START_MIN} .. {END_MIN} (minutes)")
print(f"Total spawned (attempted): {total_spawned}")
print(f"Total boarded onto trains: {total_boarded}")
print(f"Total completed arrivals: {sum(station_arrivals.values())}")
print(f"Total transfers counted: {sum(station_transfers.values())}")
print("Avg number of transfers per commuter:",
      (sum(station_transfers.values()) / sum(station_arrivals.values()))
      if sum(station_arrivals.values()) > 0 else float('nan'))

# Top stations by total handled (arrivals + transfers)
station_totals = {}
for sid in set(list(station_arrivals.keys()) + list(station_transfers.keys())):
    station_totals[sid] = station_arrivals.get(sid, 0) + station_transfers.get(sid, 0)

top10 = sorted(station_totals.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top stations (by arrivals+transfers):")
for sid, total in top10:
    # find station_name from platforms_df if available
    names = platforms_df.loc[platforms_df['station_id'].astype(int) == sid, 'station_name'].unique()
    name = names[0] if len(names) > 0 else str(sid)
    print(f" Station {name} (ID {sid}): {total}")

# Optionally save final queue sizes per station to CSV for analysis
out_rows = []
for sid in unique_station_ids:
    for line, dirs in platform_queues[sid].items():
        for dcode, q in dirs.items():
            out_rows.append({'station_id': sid, 'line': line, 'direction': dcode, 'queue_len': len(q)})

pd.DataFrame(out_rows).to_csv("data/final_platform_queues.csv", index=False)
print("Success")
