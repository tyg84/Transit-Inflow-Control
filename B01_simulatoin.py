import pandas as pd
import numpy as np


class Event:
    def __init__(self, line_id, direction_id, station_id, platform_id, train_id, event_timestamp, event_type):
        self.line_id = line_id
        self.direction_id = direction_id
        self.station_id = station_id
        self.platform_id = platform_id
        self.train_id = train_id
        self.event_timestamp = event_timestamp
        self.event_type = event_type


class Train:
    def __init__(self, train_id, capacity):
        self.train_id = train_id
        self.capacity = capacity
        self.passenger_list = []



class Platform:
    def __init__(self, line_id, direction_id, platform_id, event_timestamp, event_type):
        self.line_id = line_id
        self.passenger_list = []


class Passenger:
    def __init__(self, origin, destination, path_id):
        self.origin = origin




def generate_event_list(events):
    event_list = []
    events = events.sort_values(['event_timestamp'])
    for line_id, direction_id, station_id, platform_id, train_id, event_timestamp, event_type in zip(
            events['line_id'], events['direction_id'], events['station_id'],
            events['platform_id'], events['train_id'],
            events['event_timestamp'], events['event_type']):
        event_list.append(Event(line_id, direction_id, station_id, platform_id, train_id, event_timestamp, event_type))
    return event_list


def offload_passengers():
    ### offload passenger from the train if this is a transfer station or destination
    ## offload: move them out of the train, put to next boarding platform if transfer (event_time + transfer walk time),
    # or delete them out of the system when finish the trip record exit time (event time + walk time out).
    return


def onboard_passengers():
    ### put new passengers from entry gate to the platform. sort by arrival-at-platform time
    # onboard passenger to the train up to the capacity.
    ## update the platform queue.
    return



def simulation_main(event_list):

    for event in event_list:
        if event.train_id not in all_trains:
            all_trains[event.train_id] = Train(event.train_id, train_capacity[event.train_id])

        if event.event_type == 'Arrival':
            offload_passengers()
        elif event.event_type == 'Departure':
            onboard_passengers()




if __name__ == '__main__':
    ## generate demand first.
    all_trains = {}
    train_capacity = {}
    events = pd.read_csv('data/events.csv')
    event_list = generate_event_list(events)
    simulation_main(event_list)