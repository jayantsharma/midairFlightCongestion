import json, pickle
import requests
import os


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
    def get_route(cls, flightID):
        payload = { 'faFlightID': flightID }
        response = cls.fetch_api('DecodeFlightRoute', payload)
        response = response.json()
        if 'DecodeFlightRouteResult' in response:
            route = response['DecodeFlightRouteResult']['data']
            return route
        else:
            print('UNEXPECTED RESPONSE: {} for flightID: {}'.format(response, flightID), flush=True)


if __name__ == '__main__':
    fname = 'tracks_JFK_LAX_may17-31.pkl'
    with open(fname, 'rb') as data_dump:
        flights = pickle.load(data_dump)

    i = 0
    offset = 0
    limit = len(flights)
    succ, fail = 0, 0
    for id_, _ in list(flights.items())[offset:limit]:
        route = FlightXMLApi.get_route(id_)
        if route:
            try:
                flights[id_]['planned_route'] = route
                print(len(route))
                succ += 1
            except:
                import ipdb; ipdb.set_trace()
        else:
            fail += 1

        i += 1
        if i % 10 == 0:
            print('Complete: {}, Success: {}, Failed: {}'.format(i, succ, fail), flush=True)
            with open('tracks_JFK_LAX_may17-31.pkl', 'wb') as data_dump:
                pickle.dump(flights, data_dump)
    print('Done: {}, Success: {}, Failed: {}'.format(limit-offset, succ, fail))
    import ipdb; ipdb.set_trace()

    with open('tracks_JFK_LAX_may17-31.pkl', 'wb') as data_dump:
        pickle.dump(flights, data_dump)
