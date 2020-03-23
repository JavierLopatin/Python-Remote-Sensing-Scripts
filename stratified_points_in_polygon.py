#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:19:43 2020

@author: javier
"""

import sys
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm  # just for nice visualization


def random_points_in_polygon(shp, number=100, dist=10, maxiter=1000, n_jobs=8):
    '''
    Generate random points inside polygons

    Parameters
    ----------
    - shp: String
            Absolute path to the input shapefile (multipolygon)
    - number: Integer
            Number of points to generate inside each polygon (class)
    - dist: Integer
            distance value of the stratified sampling in meters
            default = 10
    - maxiter: Integer
            Maximum number of iterations before exit. Usefull if the polygon size
            does not hold the selected 'number' of points due to 'dist'

    '''
    def my_function(polygon):
        points = []
        min_x, min_y, max_x, max_y = polygon.bounds
        i = 0
        while len(points) < number:
            point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(point):
                if i == 0:
                    points.append(point)
                if i != 0:
                    try:
                        points.append(point)
                        dist = [point.distance(x) for x in points][:-1]
                        # delete last point if distance to other points > dist
                        if np.any(np.array(dist) < dist):
                            points.pop()
                    except i > maxiter:
                        sys.exit(1)
                i += 1

        return gpd.GeoDataFrame(crs=shp.crs, geometry=points)  # returns list of shapely point

    myList = list(shp.geometry)

    # parallel processing through the list
    num_cores = multiprocessing.cpu_count()-2
    # Parallel(n_jobs=n_jobs)(delayed(funciton)(parateters of function) for polygon in tqdm(myList))
    outlist = Parallel(n_jobs=n_jobs)(delayed(my_function)(polygon) for polygon in tqdm(myList))

    return pd.concat(outlist)


if __name__ == "__main__":

    shp = 'shapefiles/SuisunMarsh_NVCSName_diss.shp'
    shapefile = gpd.read_file(shp)

    samples = random_points_in_polygon(shapefile)

    samples.to_file('samplepoints.shp')
