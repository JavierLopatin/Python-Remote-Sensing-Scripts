#! /usr/bin/env python

#######################################################################################################
#
# ExtractValues.py
#
# A python script to extract raster values using a shapefile. 
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

import os, argparse, sys
import pandas as pd
import numpy as np

try:
   import rasterio
except ImportError:
   print("ERROR: Could not import Rasterio Python library.")
   print("Check if Rasterio is installed.")

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

def ExtractValues(raster, shp, func, ID):
    """ Extract raster values by a shapefile mask.
    Several statistics are allowed.
    """
    # Raster management
    name = os.path.basename(raster)
    r = rasterio.open(raster)
    affine = r.affine 
    array = r.read()
    bands = array.shape[0]
    bandNames = []
    for i in range(bands):
        a = "B" + str(i+1)
        bandNames.append(a)
    
    # Shapefile management
    shape = shapefile.Reader(shp)
    records = pd.DataFrame(shape.records())
    n = pd.DataFrame(shape.fields)[0].values.tolist().index(ID)
    id = records[n-1]

    # empty matrix to store results
    matrix = np.empty((len(records), bands+1), dtype=object)
    matrix[:,0] = id
    colnamesSHP = [ID]

    # Final colnames
    colNames = colnamesSHP + bandNames

    # Extract values
    for i in range(bands):
        # stats 
        array = r.read(i+1)
        stats = zonal_stats(shp, array, affine=affine, stats=func)
        x = pd.DataFrame(stats)
        matrix[:,i+1] = x[func]
    
    # set the final data frame
    df = pd.DataFrame(matrix, columns=colNames)
    return df

def ExtractPointValues(raster, shp, ID):
    from rasterstats import point_query
    """ Extract raster values by a shapefile point mask.
    """
    # Raster management
    name = os.path.basename(raster)
    r = rasterio.open(raster)
    affine = r.affine 
    array = r.read()
    bands = array.shape[0]
    bandNames = []
    for i in range(bands):
        a = "B" + str(i+1)
        bandNames.append(a)
    
    # Shapefile management
    shape = shapefile.Reader(shp)
    records = pd.DataFrame(shape.records())
    n = pd.DataFrame(shape.fields)[0].values.tolist().index(ID)
    id = records[n-1]

    # empty matrix to store results
    matrix = np.empty((len(records), bands+1), dtype=object)
    matrix[:,0] = id
    colnamesSHP = [ID]

    # Final colnames
    colNames = colnamesSHP + bandNames

    # Extract values
    for i in range(bands):
        # stats 
        array = r.read(i+1)
        stats = point_query(shp, array, affine=affine)
        x = pd.DataFrame(stats)
        matrix[:,i+1] = x[0]
    
    # set the final data frame
    df = pd.DataFrame(matrix, columns=colNames)
    return df


if __name__ == "__main__":

    # create the arguments for the algorithm
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--raster', help='Input raster', type=str)   
    parser.add_argument('-s', '--shapefile', help='Input shapefile', type=str)
    parser.add_argument('-f', '--function', help='Input function to extract [default = "mean"]', type=str, default="mean")
    parser.add_argument('-i', '--id', help='Shapefile ID to store in the CSV', type=str)
    parser.add_argument('-p','--points', help='Shapefile are points',  action="store_true", default=False)   
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())

    # run Extraction
    raster = args['raster']
    shp    = args['shapefile']
    ID     = args['id']
    func   = args['function']
    
    # Check that the input parameter has been specified.
    if args['raster'] == None:
       # Print an error message if not and exit.
       print("Error: No input image file provided.")
       sys.exit()

    if args['shapefile'] == None:
       # Print an error message if not and exit.
       print("Error: No input shapefile file provided.")
       sys.exit()

    if args['id'] == None:
       # Print an error message if not and exit.
       print("Error: No input id provided.")
       sys.exit()
    
    if args['points']==True:
        if args['function'] == None:
           # Print an error message if not and exit.
           print("Error: No extracting function provided.")           
           sys.exit()
        
        df = ExtractPointValues(raster, shp, ID)
    else:
        df = ExtractValues(raster, shp, func, ID)

    # Save to CSV file
    name = os.path.basename(shp)
    df.to_csv(name[:-4] + ".csv", index=False, heather=True, na_rep='NA') 


