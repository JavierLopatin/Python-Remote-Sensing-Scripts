# -*- coding: utf-8 -*-
"""
CloudMetric2Raster.py

Rasterize a CloudMetrics file created by FUSION

It needs: the shapefile is a grid of polygons of the size of the raster pixel

Usage:
    python CloudMetric2Raster.py -l <Imput lidar data> 
                                 -s <Imput shapefile> 
                                 -i <Imput shapefile ID>

Example:
    python CloudMetric2Raster.py -l lidar.las -s shape.shp -i ID
    
Created on Fri Mar  9 15:37:06 2018

@author: Lopatin
"""

import os, glob, math, argparse, rasterio
from subprocess import call
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

### path and name of the input shapefile
FusionDir = "C:/FUSION/"
lastoolsDir = "C:/lastools/bin/"

# create the arguments for the algorithm
parser = argparse.ArgumentParser()
parser.add_argument('-l','--lidar', help='Input Lidar data', type=str)   
parser.add_argument('-s', '--shapefile', help='Input shapefile', type=str)
parser.add_argument('-i', '--id', help='Shapefile ID to store in the CSV', type=str)
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
args = vars(parser.parse_args())

# asign variable names
lidar     = args['lidar']
input_shp = args['shapefile']
ID        = args['id']

### load shapefile
r = gpd.read_file(input_shp)
crs = r.crs # get CRS

# get the position of the ID column in the attribute table
idName = str
for i in range(len(r.columns)):
    if r.columns[i] == ID:
        idName = i
        break
    else:
        continue

# Get the ID values/names
GetIdNames = r[ID]

# create a tamporal folder to store intermediate files
if not os.path.exists("FUSION_tmp"):
    os.makedirs("FUSION_tmp")

### Create cloudmetric file
print("Creating Cloumetric file...")
for i in tqdm( range(len(r)) ):
    outname = "FUSION_tmp/"+str(GetIdNames[i])
    # Save each shapefile    
    shape = r[i:(i+1)]
    shape.to_file(outname + ".shp", driver='ESRI Shapefile')
    # Cut the point cloud using lastools
    process = lastoolsDir+"lasclip -i "+lidar+" -poly "+outname+".shp -o "+outname+".las" 
    call(process)
    # Create cloudmetrics for each .las file
    process = FusionDir+"cloudmetrics "+outname+".las"+" Cloudmetrics.csv"
    call(process)  
    # delete files from memory
    files = glob.glob('FUSION_tmp/*')
    for f in files:
        os.remove(f)  
print("Done!")

### Load the .csv file
columns = [1] + list(range(13, 51))
metrics = pd.read_csv("CloudMetrics.csv", usecols=columns)
# replace spaces by "_" in the columnames 
metrics.columns = metrics.columns.str.replace(" ", "_")
# replece the column mane 'FileTitle' with the shapefile ID
metrics = metrics.rename(columns = {'FileTitle' : ID})


### merge the metrics with the shapefile
r = r.merge(metrics, on=ID)
out_shp = input_shp[:-4]+"_metrics.shp"
# save shapefile
r.to_file(out_shp, driver='ESRI Shapefile')

### creating metric raster
# load the shapefile (important as the column amnes may have been shortened)
r = gpd.read_file(out_shp)
names = r.columns[-39:] # get column names
if 'geometry' in names:
    names = names[:-1]
pixel_size = pixel_size = math.sqrt(r.area[0]) # get pixel size
crs = r.crs # set CRS

### rasterize all metrics
print("Rasterizing the metrics...")
for i in tqdm( range(len(names)) ):
    process = "gdal_rasterize -a "+names[i]+" -tr "+str(pixel_size)+" "+str(pixel_size)+" -l "+out_shp[:-4]+" "+out_shp+" "+"FUSION_tmp/"+names[i]+".tif"
    call(process)  
print("Done!")

### Stack rasters
# Read metadata of first file
with rasterio.open("FUSION_tmp/"+names[0]+".tif") as src0:
    meta = src0.meta

# Update meta to reflect the number of layers
meta.update(count = len(names))

# Read each layer and write it to stack
outName = input_shp[:-4]+"_metrics.tif"
with rasterio.open(outName, 'w', **meta) as dst:
    for id, layer in enumerate(names):
        with rasterio.open("FUSION_tmp/"+layer+".tif") as src1:
            dst.write_band(id + 1, src1.read(1))

# add the CRS to raster
with rasterio.open(outName, 'r+') as r:
    r.crs = crs

