#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C
Rasterize all shapefile columns into a multiband raster

reated on Sat Dec  5 11:44:50 2020

@author: Javier Lopatin
"""

import geopandas as gpd
from geocube.api.core import make_geocube
import argparse

# create the arguments for the algorithm
parser = argparse.ArgumentParser()

parser.add_argument('-i','--inputShapefile',
  help='Input raster', type=str, required=True)
parser.add_argument('-r','--resolution',
  help='Output resolution', type=float)
parser.add_argument('-p','--preprop',
  help='Resolution in a tuple, e.g. (-5, 5) for a 5X5 m pixels.',
  action="store_true", default=False)
 
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
args = vars(parser.parse_args())

# data imputps/outputs
inData = args["inputShapefile"]
res = args["resolution"]
outraster = inData[:-4]+'_raster.tif'

# load shapefile
s = gpd.read_file(inData)

# rasterize
print("Rasterizing...")
out_grid = make_geocube(vector_data=s, resolution=(-res, res))
print('Done!')
# save to disk
print('Saving to disk...')
out_grid.rio.to_raster(outraster)
print('Done!')
