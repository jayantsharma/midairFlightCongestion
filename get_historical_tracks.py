import json
import requests
import os
# import log


class FlightXMLApi:
    fxmlUrl = "https://flightxml.flightaware.com/json/FlightXML2/"
    username = os.environ['FLIGHT_XML_USERNAME']
    api_key = os.environ['FLIGHT_XML_APIKEY']


    @classmethod
    def fetch_api(cls, query, payload):
        response = requests.get(cls.fxmlUrl + query, params=payload, auth=(cls.username, cls.api_key))
        if response.status_code != 200:
            print('ERROR: {} for {} with payload: {}'.format(response.status_code, query, payload), flush=True)
            return
        return response
    

    @classmethod
    def fetch_track(cls, ident, departure_time):
        payload = { 'faFlightID': '{}@{}'.format(ident, departure_time) }
        response = cls.fetch_api('GetHistoricalTrack', payload)
        if response:
            response = response.json()
            if 'GetHistoricalTrackResult' in response:
                res = response['GetHistoricalTrackResult']['data']
                return res
            else:
                print('UNEXPECTED RESPONSE: {} for ident: {}, departure_time: {}'.format(response, ident, departure_time), flush=True)


    @classmethod
    def get_track(cls, flight):
        departure_time = flight['departuretime']
        if flight['actual_ident']:
            ident = flight['actual_ident']
            track = cls.fetch_track(ident, departure_time)
            if track:
                return track

        ident = flight['ident']
        return cls.fetch_track(ident, departure_time)


if __name__ == '__main__':
    with open('airlineFlightSchedule_JFK_LAX.json', 'r') as jsonf:
        flights = json.load(jsonf)

    offset = 120
    limit = len(flights)
    succ, fail = 0, 0
    for i, flight in enumerate(flights[offset:limit]):
        track = FlightXMLApi.get_track(flight)
        if track:
            flight['track'] = track
            succ += 1
        else:
            fail += 1

        if (i+1) % 10 == 0:
            print('Complete: {}, Success: {}, Failed: {}'.format(i+1, succ, fail), flush=True)
    print('Done: {}, Success: {}, Failed: {}'.format(limit-offset, succ, fail))

    with open('airlineFlightSchedule_JFK_LAX.json', 'w') as jsonf:
        flights = json.dump(flights, jsonf)
