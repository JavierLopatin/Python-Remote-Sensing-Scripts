#! /usr/bin/env python

#######################################################################################################
#
# ExtractValues.py
#
# A python script to extract raster values using a shapefile and parralel proessing.
# Results are stored in a CSV file
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Date: 09/08/2016
# Version: 1.0
#
# Info: Several statistics are allowed when applied to polygon shapefiles, including:
#   - min
#   - max
#   - mean [default]
#   - count
#   - sum
#   - std
#   - median
#   - majority
#   - minority
#   - unique
#   - range
#   - nodata
#   - percentile
#
# Usage:
#
# python ExtractValues.py -r <Imput raster> -s <Imput shapefile>
#                         -i <Imput function> -s <Shapefile ID> -p <If shp are points>
#
#  Examples for polygon shapefile: python ExtractValues.py -r raster.tif -s shape.shp -i ID
#                                  python ExtractValues.py -r raster.tif -s shape.shp -f median -i ID
#
#  Examples for points shapefile: python ExtractValues.py -r raster.tif -s shape.shp -i ID -p
#
########################################################################################################

import os
import argparse
import sys
import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm 
import rasterio

try:
    from rasterstats import zonal_stats
except ImportError:
    print("ERROR: Could not import rasterstats Python library.")
    print("Check if rasterstats is installed.")

try:
    import shapefile
except ImportError:
    print("ERROR: Could not import PyShp Python library.")
    print("Check if PyShp is installed.")


def ExtractValues(raster, shp, func, ID, num_cores, points):
    """ Extract raster values by a shapefile mask.
    Several statistics are allowed.
    """
    # read raster properties
    with rasterio.open(raster) as r:
        count = r.count
    
    # band names
    bandNames = []
    for i in range(count):
        a = "B" + str(i+1)
        bandNames.append(a)
        
    # Shapefile management
    shape = shapefile.Reader(shp)
    records = pd.DataFrame(shape.records())
    n = pd.DataFrame(shape.fields)[0].values.tolist().index(ID)
    id = records[n-1]
    
    # Extract values
    if points == True:
        def _function(i, shp, raster):
            stats = point_query(shp, raster, band=i+1)
            return pd.DataFrame(stats)
    else:
        def _function(i, shp, raster):
            stats = zonal_stats(shp, raster, stats=func, band=i+1)
            return pd.DataFrame(stats)
        
    # parallel processing
    stats = Parallel(n_jobs=num_cores)(delayed(_function)(i, shp, raster)
                                       for i in tqdm(range(count)))
    # set the final data frame
    df = pd.concat(stats, axis=1)  # concatenate all dataframes into one
    df.columns = bandNames  # add colum names
    df.index = id  # and shapefile ID to index
    # save data to .CSV file
    name = os.path.basename(raster)
    name2 = os.path.basename(shp)
    df.to_csv(name[:-4] + "_" + name2[:-4] +".csv", index=False, header=True, na_rep='NA')




if __name__ == "__main__":

    # create the arguments for the algorithm
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--raster', help='Input raster', type=str)
    parser.add_argument('-s', '--shapefile', help='Input shapefile', type=str)
    parser.add_argument(
        '-f', '--function', help='Input function to extract [default = "mean"]', type=str, default="mean")
    parser.add_argument('-i', '--id', help='Shapefile ID to store in the CSV', type=str)
    parser.add_argument('-p', '--points', help='Shapefile are points',
                        action="store_true", default=False)
    
    parser.add_argument('-c', '--cores', help='Number of cores in parallel processing',
                        type=int, default=4)
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())

    # run Extraction
    raster    = args['raster']
    shp       = args['shapefile']
    ID        = args['id']
    func      = args['function']
    num_cores = args['cores']
    points    = args['points']

    # Check that the input parameter has been specified.
    if raster == None:
        # Print an error message if not and exit.
        print("Error: No input image file provided.")
        sys.exit()

    if shp == None:
        # Print an error message if not and exit.
        print("Error: No input shapefile file provided.")
        sys.exit()

    if ID == None:
        # Print an error message if not and exit.
        print("Error: No input id provided.")
        sys.exit()
        
    if func == None:
        # Print an error message if not and exit.
        print("Error: No extracting function provided.")
        sys.exit()

    if points == True:
        ExtractValues(raster, shp, ID, num_cores, points)
    else:
        ExtractValues(raster, shp, func, ID, num_cores, points)
