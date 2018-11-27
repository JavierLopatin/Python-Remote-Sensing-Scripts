#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clipping a raster image with a shapefile

Usage: python clip.py -r raster.tif -s shapefile.shp

Created on Tue Oct 23 16:32:37 2018

@author: Javier Lopatin | javier.lopatin@kit.edu
"""

import rasterio, rasterio.mask, argparse

# create the arguments for the algorithm
parser = argparse.ArgumentParser()

# set arguments
parser.add_argument('-r','--inputRaster', help='Input raster', type=str, required=True)
parser.add_argument('-v','--inputVector', help='Input raster', type=str, required=True)
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
args = vars(parser.parse_args())


# set argument
raster = args["inputRaster"]
shp = args["inputVector"]

# open and crop raster
print("Clipping raster...")
with rasterio.open(raster) as src:
     img, transform = rasterio.mask.mask(src, shp, crop=True)
     meta = src.meta.copy()
     baseName = src.name

# output name
output = baseName[:-4] + "_mask.tif"

# save
with rasterio.open(output, 'w', **meta) as dst:
    dst.write(img, 1)
print("Done!")
