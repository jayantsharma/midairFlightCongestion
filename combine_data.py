import pickle
from glob import glob


data_dump = 'tracks_JFK_LAX_may17-31.pkl'
flights = pickle.load(open(data_dump, 'rb'))

part_files = glob('tracks_JFK_LAX_may17-31_*.pkl')
for part_file in part_files:
    part_data = pickle.load(open(part_file, 'rb'))
    for ident, flight_data in part_data.items():
        if 'shortest_path_cost' in flight_data:
            flights[ident]['shortest_path_cost'] = flight_data['shortest_path_cost']
            flights[ident]['shortest_path_details'] = flight_data['shortest_path_details']


shortest_paths_found = sum(1 for _, flt in flights.items() if 'shortest_path_cost' in flt)
print(shortest_paths_found)
assert shortest_paths_found == 343

pickle.dump(flights, open(data_dump, 'wb'))
