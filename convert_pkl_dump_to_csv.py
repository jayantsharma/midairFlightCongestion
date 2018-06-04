import csv
import pickle
from datetime import datetime


flights = pickle.load(open('tracks_JFK_LAX_may17-31.pkl', 'rb'))

with open('simulation.csv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['flightID', 'departuretime', 'arrivaltime', 'planned_path_cost', 'actual_path_cost', 'shortest_path_cost'])
    for ident, flight in flights.items():
        csv_writer.writerow([
            ident, 
            datetime.fromtimestamp(flight['departuretime']).strftime('%Y:%m:%d %H:%M:%S'), 
            datetime.fromtimestamp(flight['arrivaltime']).strftime('%Y:%m:%d %H:%M:%S'), 
            flight.get('planned_path_cost'), 
            flight['actual_path_cost'], 
            flight.get('shortest_path_cost')
            ])
