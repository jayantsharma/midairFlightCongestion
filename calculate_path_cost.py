import heapq
import pygrib
import numpy as np
from math import sqrt
import random
from progress.bar import Bar
import json, pickle
from datetime import datetime
from datetime import timedelta

import ipdb
import sys
import os
from subprocess import call 

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from shortest_path import JFK, LAX, grid_label, step_cost, neighbors, haversine_distance, extract_shortest_path


GRIBFILES_DIR = 'weather_data/hrrr'
PREFIX = 'https://pando-rgw01.chpc.utah.edu/hrrr/prs/'
SUFFIX = '.wrfprsf00.grib2'
LEVEL = 250


def read_gribfile(grib_fname, level=LEVEL):
    myfile = pygrib.open(grib_fname)

    try:
        u_grb = myfile.select(name='U component of wind', typeOfLevel='isobaricInhPa', level=level)[0]
    except:
        import ipdb; ipdb.set_trace()
    # Shape - (1059, 1799)
    u_wnd_matrix = u_grb.values
    lats, lons = u_grb.latlons()
    assert lats.shape == (1059, 1799)

    v_grb = myfile.select(name='V component of wind', typeOfLevel='isobaricInhPa', level=level)[0]
    v_wnd_matrix = v_grb.values
    lats1, lons1 = v_grb.latlons()

    assert (lats == lats1).sum() == np.size(lats)
    assert (lons == lons1).sum() == np.size(lons)

    return u_wnd_matrix, v_wnd_matrix, lats, lons


def distance(m, c, x, y):
    if m == float('inf'):
        return np.abs(x-c)
    else:
        return np.abs(y - m*x - c) / sqrt(m ** 2 + 1)


def breakdown_and_compute_cost(segment, shape, u_wnd_matrix, v_wnd_matrix, lats, lons):
    orig, orig_grid_label = segment[0]
    dest, dest_grid_label = segment[1]

    direction_vec = np.array((dest[0]-orig[0], dest[1]-orig[1]))
    if np.linalg.norm(direction_vec) == 0:
        # repeat points, zero cost obvs
        return 0, 0
    direction_vec = direction_vec / np.linalg.norm(direction_vec)
    
    # line of motion params
    if not direction_vec[0] == 0:
        m = direction_vec[1] / direction_vec[0]
        c = orig[1] - m * orig[0]
    else:
        m = float('inf')
        c = orig[0]         # m = inf => theta = pi/2 => x - c = 0

    path = [ orig_grid_label ]
    """
    Do a step_cost calc using coordinates first
    Then normal iterative procedure
    Last cost again using coords
    """
    nbrs = neighbors(orig_grid_label, shape)
    if dest_grid_label in nbrs:
        total_cost = step_cost(orig, dest, u_wnd_matrix, v_wnd_matrix, lats, lons, orig_grid_label, dest_grid_label)
        path.append(dest_grid_label)
        # if random.random() < .2:
        #     print('Path: {}, COST = {}'.format(' -> '.join(map(str, path)), total_cost))
        return total_cost, len(path)
    min_dist = float('inf')
    move_to = None
    for nbr in nbrs:
        movement_vec = np.array((lons[nbr]-orig[0], lats[nbr]-orig[1]))
        if np.dot(movement_vec, direction_vec) > 0:
            dist = distance(m, c, lons[nbr], lats[nbr])
            if dist < min_dist:
                move_to = nbr
                min_dist = dist
    total_cost = step_cost(orig, (lons[move_to], lats[move_to]), u_wnd_matrix, v_wnd_matrix, lats, lons, orig_grid_label, move_to)
    u = move_to
    path.append(move_to)

    while True:
        nbrs = neighbors(u, shape)
        if dest_grid_label in nbrs:
            total_cost += step_cost((lons[u], lats[u]), dest, u_wnd_matrix, v_wnd_matrix, lats, lons, u, dest_grid_label)
            path.append(dest_grid_label)
            # if random.random() < .2:
            #     print('Path: {}, COST = {}'.format(' -> '.join(map(str, path)), total_cost))
            return total_cost, len(path)
        min_dist = float('inf')
        move_to = None
        for nbr in nbrs:
            movement_vec = np.array((lons[nbr]-lons[u], lats[nbr]-lats[u]))
            if np.dot(movement_vec, direction_vec) > 0:
                dist = distance(m, c, lons[nbr], lats[nbr])
                if dist < min_dist:
                    move_to = nbr
                    min_dist = dist
        total_cost += step_cost(u, move_to, u_wnd_matrix, v_wnd_matrix, lats, lons)
        u = move_to
        path.append(move_to)


def path_cost(path, u_wnd_matrix, v_wnd_matrix, lats, lons):
    return sum(step_cost(u, nbr, u_wnd_matrix, v_wnd_matrix, lats, lons) for (u,nbr) in zip(path[:-1], path[1:]))


def download_and_get_gribfile(midflighttime):
    # Link of the type : https://pando-rgw01.chpc.utah.edu/hrrr/sfc/20180213/hrrr.t01z.wrfsfcf00.grib2
    dt = midflighttime.strftime('%Y%m%d')
    hr = round(midflighttime.hour + midflighttime.minute/60)
    if hr == 24:
        hr = 0
        dt = (midflighttime + timedelta(days=1)).strftime('%Y%m%d')
    hr = 't{:02d}z'.format(hr)
    link = PREFIX + dt + '/hrrr.' + hr + SUFFIX

    dir_ = GRIBFILES_DIR + '/{}'.format(dt)
    try:
        os.listdir(dir_)
    except:
        os.makedirs(dir_)
    fname = '{}.grib2'.format(hr)
    loc = dir_ + '/' + fname

    if not os.path.exists(loc):
        call(['wget', '-O', loc, link])

    return loc


def calculate_actual_path_cost(flight):
    track = flight['track']
    midflighttime = (flight['departuretime'] + flight['arrivaltime']) / 2
    midflighttime = datetime.fromtimestamp(midflighttime)

    grib_file = download_and_get_gribfile(midflighttime)
    u_wnd_matrix, v_wnd_matrix, lats, lons = read_gribfile(grib_file)
    shape = u_wnd_matrix.shape
    # import ipdb; ipdb.set_trace()
    
    waypoints = [ ((waypoint['longitude'], waypoint['latitude'])) for waypoint in track ]
    # reverse lat/lon before passing as arg
    waypoints = [ (pt, grid_label((pt[1], pt[0]), lats, lons)) for pt in waypoints ]
    print('GRID placement complete')
    # makes slope calculations easier
    segments = zip(waypoints[:-1], waypoints[1:])
    total_path_cost = 0
    path_lengths = []
    # bar = Bar('Processing', max = len(waypoints)-1, suffix='%(percent).1f%% - %(eta)ds')
    for segment in segments:
        path_cost, path_length = breakdown_and_compute_cost(segment, shape, u_wnd_matrix, v_wnd_matrix, lats, lons)
        total_path_cost += path_cost
        path_lengths.append(path_length)
        # bar.next()
    # bar.finish()
    print(total_path_cost)
    return total_path_cost
    # print(sum(path_lengths)-len(waypoints), grid_label(LAX, lats, lons), grid_label(JFK, lats, lons))

    # departuretime = flight['departuretime']
    # departuretime = datetime.fromtimestamp(departuretime) - timedelta(hours=5)
    # arrivaltime = flight['arrivaltime']
    # arrivaltime = datetime.fromtimestamp(arrivaltime) - timedelta(hours=5)
    # time_taken = arrivaltime - departuretime
    # print('Actual time taken: {}'.format(time_taken.seconds))

    # m = Basemap(llcrnrlon = -130,llcrnrlat = 20, urcrnrlon = -70,
    #            urcrnrlat = 52 , projection = 'merc', area_thresh =10000 ,
    #            resolution='l')
    # m.drawcoastlines()
    # m.drawcountries()
    # m.fillcontinents(color = 'coral')
    # m.drawmapboundary()
    # my_p = m.drawparallels(np.arange(20,80,4),labels=[1,1,0,0])
    # my_m = m.drawmeridians(np.arange(-140,-60,4),labels=[0,0,0,1])


    # points = [ pt for (pt,_) in waypoints ]
    # lons = [ pt[0] for pt in points ]
    # lats = [ pt[1] for pt in points ]
    # # plot airports
    # x,y = m([JFK[1], LAX[1]], [JFK[0], LAX[0]])
    # m.plot(x, y, 'ro', markersize=10)
    # # plot route
    # x,y = m(lons, lats)
    # m.plot(x, y, 'bo', markersize=1)

    # # day = grib_fname.split('_')[0]
    # # time = grib_fname.split('.')[1]
    # # plt.savefig('flight_path_{}_{}.png'.format(day, time), bbox_inches='tight')
    # plt.show()


def calculate_shortest_path(flight):
    midflighttime = (flight['departuretime'] + flight['arrivaltime']) / 2
    midflighttime = datetime.fromtimestamp(midflighttime)
    grib_file = download_and_get_gribfile(midflighttime)

    cost, lats, lons = extract_shortest_path(grib_file, level=LEVEL)
    return cost, lats, lons


if __name__ == '__main__':
    # 15 Feb '18 - Hour 12 Forecast 00
    # grib_fname = 'feb12_hrrr.t12z.wrfsfcf00.grib2'
    data_dump = 'tracks_JFK_LAX_may17-31.pkl'
    flights = pickle.load(open(data_dump, 'rb'))
    print('Load JSON complete')
    res = {}
    i = 0

    mul = 3
    inc = 86
    offset = mul * inc
    limit = min(offset + inc, len(flights))
    data_dump = 'tracks_JFK_LAX_may17-31_{}-{}.pkl'.format(offset,limit)

    for id_, flight in list(flights.items())[offset:limit]:
        # cost = calculate_actual_path_cost(flight)
        cost, lats, lons = calculate_shortest_path(flight)

        flights[id_]['shortest_path_cost'] = cost
        flights[id_]['shortest_path_details'] = { 'lats': lats, 'lons': lons }

        i += 1
        if i % 1 == 0:
            print('{}/{} DONE'.format(i, limit - offset))
            pickle.dump(flights, open(data_dump, 'wb'))
            print('PKL written')
    pickle.dump(flights, open(data_dump, 'wb'))
