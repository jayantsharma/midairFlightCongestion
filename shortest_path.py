import heapq
import pygrib
import numpy as np
from math import sqrt
from progress.bar import Bar
import ipdb
import sys

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

JFK = (40.641311100, -73.778139100)
LAX = (33.950001300, -118.386123200)
AIRPLANE_SPEED = 205        # 400 knots in m/s
# MOVEMENT_UNIT_COST = 1/AIRPLANE_SPEED


def neighbors(u, shape):
    lat_idx, lon_idx = u
    candidates = [ (lat_idx + i, lon_idx + j) for i in range(-1,2) for j in range(-1,2) ]
    neighbors = [ (v,w) for (v,w) in candidates if u != (v,w) and 0 <= v < shape[0] and 0 <= w < shape[1] ]
    return neighbors


# return Haversine Distance in meters
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    a = np.square(np.sin((lat1 - lat2)/2)) + np.cos(lat1) * np.cos(lat2) * np.square(np.sin((lon1 - lon2)/2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    R = 6371 * 1000
    d = R * c
    return d


def step_cost(u, nbr, u_wnd_matrix, v_wnd_matrix, lats, lons, u_grid_label=None, nbr_grid_label=None):
    if not u_grid_label:        
        # u itself is the grid_label, not coords
        # Intended direction of movement - center -> center
        heading_vec = np.array((lons[nbr]-lons[u], lats[nbr]-lats[u]))
        heading_unit_vec = heading_vec / np.linalg.norm(heading_vec)
        heading_distance = haversine_distance(lats[nbr], lons[nbr], lats[u], lons[u])
    else:                       
        # u has coords : (lon, lat) coz its easier to interpret as (x,y)
        heading_vec = np.array((nbr[0]-u[0], nbr[1]-u[1]))
        heading_unit_vec = heading_vec / np.linalg.norm(heading_vec)
        heading_distance = haversine_distance(nbr[1], nbr[0], u[1], u[0])
        u = u_grid_label
        nbr = nbr_grid_label

    # Wind vector gotten by averaging across the 2 cells
    u_wnd = (u_wnd_matrix[u] + u_wnd_matrix[nbr])/2.
    v_wnd = (v_wnd_matrix[u] + v_wnd_matrix[nbr])/2.
    wind_vec = np.array((u_wnd, v_wnd))
    wind_unit_vec = wind_vec / np.linalg.norm(wind_vec)

    if heading_unit_vec[0] != 0:    # division by zero
        tan_phi = heading_unit_vec[1] / heading_unit_vec[0]
        a = 1 + tan_phi ** 2
        d = (v_wnd/AIRPLANE_SPEED) - (u_wnd/AIRPLANE_SPEED) * tan_phi
        b = 2 * d
        c = d ** 2 - tan_phi ** 2

        sines = []
        disc = sqrt(b ** 2 - 4 * a * c)
        sine = (-b + disc) / (2*a)
        if -1 <= sine <= 1:
            sines.append(sine)
        sine = (-b - disc) / (2*a)
        if -1 <= sine <= 1 and disc > 0:    # else, this is just a duplicate
            sines.append(sine)

        max_comp = float('-inf')
        sol_heading_speed = None
        for sine in sines:
            cos = sqrt(1 - sine ** 2)
            for mul in (-1,1):
                cos_ = cos * mul
                movement_vec = AIRPLANE_SPEED * np.array((cos_, sine))
                heading_vec_= movement_vec + wind_vec
                heading_speed = np.linalg.norm(heading_vec_)
                comp = np.dot(heading_vec_ / heading_speed, heading_unit_vec)
                if comp > max_comp:
                    max_comp = comp
                    sol_heading_speed = heading_speed

    else:
        cos = - u_wnd / AIRPLANE_SPEED
        max_comp = float('-inf')
        sol_heading_speed = None
        sin = sqrt(1 - cos ** 2)
        for mul in (-1,1):
            sin_ = sin * mul
            movement_vec = AIRPLANE_SPEED * np.array((cos, sin_))
            heading_vec_= movement_vec + wind_vec
            heading_speed = np.linalg.norm(heading_vec_)
            comp = np.dot(heading_vec_ / heading_speed, heading_unit_vec)
            if comp > max_comp:
                max_comp = comp
                sol_heading_speed = heading_speed
    
    try:
        assert 1 - max_comp < 1e-8
    except:
        import ipdb; ipdb.set_trace()

    # The way formulae have been written, all costs should be in seconds
    wt = heading_distance / sol_heading_speed
    return wt


    ### GRID SEARCH for theta ###
    # min_res = float('inf')
    # for theta in np.arange(-np.pi, np.pi, .1):
        # j_comp = airplane_speed * np.sin(theta) + v_wnd
        # i_comp = airplane_speed * np.cos(theta) + u_wnd
        # theta = np.arctan2(j_comp, i_comp)
        # res = np.abs(theta - phi)
        # if res < min_res:
            # min_res = res
            # min_theta = theta

    # movement_unit_vec = np.array((np.cos(min_theta), np.sin(min_theta)))
    # movement_vec = airplane_speed * movement_unit_vec
    # heading_vec_ = movement_vec + wind_vec
    # try:
        # assert np.dot(heading_vec_, heading_unit_vec) > 0
    # except AssertionError:
        # import ipdb; ipdb.set_trace()
    # heading_speed = np.linalg.norm(heading_vec_)


    ### OLD METHOD - crap ###
    ## Movement + Wind = Heading (Relative Motion!!)
    # movement_unit_vec = heading_unit_vec - wind_unit_vec
    # movement_unit_vec = movement_unit_vec / np.linalg.norm(movement_unit_vec)
    ## How many meters would I move in movement direction to move 1 meter in heading direction ?
    # num_movement_units_per_heading_unit = 1 / np.dot(movement_unit_vec, heading_unit_vec)
    # num_movement_units = heading_distance * num_movement_units_per_heading_unit
    # movement_cost = num_movement_units * movement_unit_cost

    ## Cost = movement_cost + wind_resistance_cost
    ## wind_resistance_cost = resistance offered in direction of heading = - dot_product
    # wind_resistance_cost_per_unit_movement = -np.dot(movement_unit_vec, wind_vec)
    # wind_resistance_cost = wind_resistance_cost_per_unit_movement * num_movement_units

    # wt = movement_cost + wind_resistance_cost


def shortest_path(s,t,u_wnd_matrix,v_wnd_matrix,lats,lons):
    shape = u_wnd_matrix.shape
    num_rows, num_cols = shape

    # keep track of distances
    dist = np.full(shape, float('inf'))
    dist[s] = 0

    # Cost of moving a unit distance
    # Take avg airplane speed to be 400 knots ~ 205 m/s
    # unit_cost (time taken to move 1 meter) = 1/205

    # keep track of parents 
    parents = [ [ None ] * num_cols for _ in range(num_rows) ]

    # make waiting easy
    bar = Bar('Processing', max = lats.shape[0] * lats.shape[1], suffix='%(percent).1f%% - %(eta)ds')

    heap = []
    heapq.heappush(heap, (dist[s],s))
    reached = set()
    while True:
        d, u = heapq.heappop(heap)
        if u not in reached:
            if u == t:
                path = []
                while t:
                    path.append(t)
                    t = parents[t[0]][t[1]]
                bar.finish()
                return d, path[::-1]
            reached.add(u)
            bar.next()
            # if len(reached) % 50 == 0:  print(len(reached), min(reached, key=lambda x: x[0] ** 2 + x[1] ** 2))
            nbrs = neighbors(u, shape)
            for nbr in nbrs:
                if nbr not in reached:
                    wt = step_cost(u, nbr, u_wnd_matrix, v_wnd_matrix, lats, lons)
                    d_nbr = dist[u] + wt
                    if dist[nbr] > d_nbr:
                        dist[nbr] = d_nbr
                        heapq.heappush(heap, (d_nbr, nbr))
                        parents[nbr[0]][nbr[1]] = u


def grid_label(coords, lats, lons):
    num_cols = lats.shape[1]
    lat, lon = coords
    flattened_idx = np.argmin((lats - lat) ** 2 + (lons - lon) ** 2)
    idx = divmod(flattened_idx, num_cols)
    return idx


def extract_shortest_path(fname, level=850):
    myfile = pygrib.open(fname)

    u_grb = myfile.select(name='U component of wind', typeOfLevel='isobaricInhPa', level=level)[0]
    # Shape - (1059, 1799)
    u_wnd_matrix = u_grb.values
    lats, lons = u_grb.latlons()

    v_grb = myfile.select(name='V component of wind', typeOfLevel='isobaricInhPa', level=level)[0]
    v_wnd_matrix = v_grb.values
    lats1, lons1 = v_grb.latlons()

    assert (lats == lats1).sum() == np.size(lats)
    assert (lons == lons1).sum() == np.size(lons)

    orig_coords = JFK
    s = grid_label(orig_coords, lats, lons)
    print('New York : {}'.format(s))
    dest_coords = LAX
    t = grid_label(dest_coords, lats, lons)
    print('Los Angeles : {}'.format(t))

    dist, path = shortest_path(s, t, u_wnd_matrix, v_wnd_matrix, lats, lons)
    lats = np.array([lats[p] for p in path])
    lons = np.array([lons[p] for p in path])
    return dist, lats, lons


if __name__ == '__main__':
    # 15 Feb '18 - Hour 12 Forecast 00
    grib_fname = sys.argv[1]
    d, lats, lons = extract_shortest_path(grib_fname)
    print(d, len(lats))

    m = Basemap(llcrnrlon = -130,llcrnrlat = 20, urcrnrlon = -70,
               urcrnrlat = 52 , projection = 'merc', area_thresh =10000 ,
               resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color = 'coral')
    m.drawmapboundary()
    my_p = m.drawparallels(np.arange(20,80,4),labels=[1,1,0,0])
    my_m = m.drawmeridians(np.arange(-140,-60,4),labels=[0,0,0,1])


    # plot airports
    x,y = m([JFK[1], LAX[1]], [JFK[0], LAX[0]])
    m.plot(x, y, 'ro', markersize=10)
    # plot route
    x,y = m(lons, lats)
    m.plot(x, y, 'bo', markersize=1)

    day = grib_fname.split('_')[0]
    time = grib_fname.split('.')[1]
    plt.savefig('flight_path_{}_{}.png'.format(day, time), bbox_inches='tight')
    plt.show()

    ipdb.set_trace()
