#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dissolve all vector fields

Usage: python Dissolve_all.py -i vector.shp

@author: javier
"""

import argparse
import geopandas as gpd

# create the arguments for the algorithm
parser = argparse.ArgumentParser()

# set arguments
parser.add_argument('-i','--inputShapefile', help='Input shapefile', type=str, required=True)
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
args = vars(parser.parse_args())

# set argument
shp = args["inputShapefile"]

vector = gpd.read_file(shp)

def dissolve_all(vector):
    # add dummy field to perform the dissolve
    vector.loc[:, "dissolve"] = 1
    # perform dissolve
    print("Dissolving vector file...")
    return vector.dissolve(by = "dissolve")

vector =  dissolve_all(shp):

# save new vector
vector.to_file(shp[:-4] + "_diss.shp")
print("Done!")
