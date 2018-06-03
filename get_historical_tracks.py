import json
import requests
import os


class FlightXMLApi:
    fxmlUrl = "https://flightxml.flightaware.com/json/FlightXML2/"
    username = os.environ['FLIGHT_XML_USERNAME']
    api_key = os.environ['FLIGHT_XML_APIKEY']
    seen = set()


    @classmethod
    def fetch_api(cls, query, payload):
        response = requests.get(cls.fxmlUrl + query, params=payload, auth=(cls.username, cls.api_key))
        if response.status_code != 200:
            print('ERROR: {} for {} with payload: {}'.format(response.status_code, query, payload), flush=True)
            return
        return response
    

    @classmethod
    def fetch_track(cls, ident, departure_time):
        flightID = cls.fetch_api('GetFlightID', { 'ident': ident, 'departureTime': departure_time }).json().get('GetFlightIDResult')
        if not flightID or flightID in cls.seen:
            return

        # payload = { 'faFlightID': '{}@{}'.format(ident, departure_time) }
        cls.seen.add(flightID)
        payload = { 'faFlightID': flightID }
        response = cls.fetch_api('GetHistoricalTrack', payload)
        if response:
            response = response.json()
            if 'GetHistoricalTrackResult' in response:
                track = response['GetHistoricalTrackResult']['data']
                return (flightID, track)
            else:
                print('UNEXPECTED RESPONSE: {} for ident: {}, departure_time: {}'.format(response, ident, departure_time), flush=True)


    @classmethod
    def get_track(cls, flight):
        ident = flight['ident']
        departure_time = flight['departuretime']
        return cls.fetch_track(ident, departure_time)


if __name__ == '__main__':
    fname = 'flights_JFK_LAX_may17-31.json'
    with open(fname) as jsonf:
        flights = json.load(jsonf)

    offset = 0
    limit = len(flights)
    succ, fail = 0, 0
    res = {}
    for i, flight in enumerate(flights[offset:limit]):
        res_ = FlightXMLApi.get_track(flight)
        if res_:
            flightID, track = res_
            res[flightID] = { 
                    'ident': flight['ident'],
                    'departuretime': flight['departuretime'],
                    'track': track
                    }
            # print(flight['ident'], flight['departuretime'], flightID, len(track))
            succ += 1
        else:
            fail += 1

        if (i+1) % 10 == 0:
            print('Complete: {}, Success: {}, Failed: {}'.format(i+1, succ, fail), flush=True)
    print('Done: {}, Success: {}, Failed: {}'.format(limit-offset, succ, fail))
    import ipdb; ipdb.set_trace()

    with open('tracks_JFK_LAX_may17-31.json', 'w') as jsonf:
        flights = json.dump(res, jsonf)
