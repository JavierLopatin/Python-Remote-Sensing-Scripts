#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dissolve vector by attribute and optionally delete small objects

parameters:
    -i: input vector [str; require]
    -a: attribute to use for the dissove [str; require]
    -e: size (square meters) of the small objects to eliminate [int; optional]

Example: python Dissolve.py -i vector.shp -a ID -e 2500

@author: Javier Lopatin | javier.lopatin@kit.edu
"""

import argparse
import geopandas as gpd

# create the arguments for the algorithm
parser = argparse.ArgumentParser()

# set arguments
parser.add_argument('-i','--inputVector', help='Input raster', type=str, required=True)
parser.add_argument('-a','--Attribute', help='Attribute to use', type=str, required=True)
parser.add_argument('-e','--Eliminate', help='Eliminate objects bellow size (square meters)', type=int, required=False)
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
args = vars(parser.parse_args())

# set argument
shp = args["inputVector"]
attribute = args["Attribute"]
eliminate = args["Eliminate"]

# read vector file
vector = gpd.read_file(shp)
# if elimination of small attributes is True:
print('Eliminating small objects...')
if (eliminate == True):
    mask = vector.area > eliminate
    vector = vector.loc[mask]
# perform dissolve
print("Dissolving vector file...")
vector = vector.dissolve(by = attribute)
# save new vector
vector.to_file(shp[:-4] + "_diss.shp")
print("Done!")
