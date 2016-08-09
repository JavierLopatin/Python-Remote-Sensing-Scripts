#! /usr/bin/env python

#########################################################################
#
# MNF_cmd.py
# A python script to perform Brigtness Normalization of hyperspectral data
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Date: 09/08/2016
# Version: 1.0
#
# Usage:
#
# python MNF.py <Imput raster format> 
#
# examples:    python BrightnessNormalization_cmd.py -f tif 
#            
#
#########################################################################


import os, glob, argparse
import numpy as np
try:
   import rasterio
except ImportError:
   print("ERROR: Could not import Rasterio Python library.")
   print("Check if Rasterio is installed.")

## Functions 

def BrigthnessNormalization(img):
    bn = img / np.sqrt( np.sum((img**2), 0) )
    return bn

def saveImage(img, inputRaster):
    # Save TIF image to a nre directory of name MNF
    output = "BN/" + name[:-4] + "_BN.tif"
    new_dataset = rasterio.open(output, 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=int(img.shape[0]), dtype=inputRaster.dtypes[0],
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(img)
    new_dataset.close()

### Run process
        
if __name__ == "__main__":

# create the arguments for the algorithm
    parser = argparse.ArgumentParser()

    parser.add_argument('-f','--format', help='Imput raster format, e.g: tif', type=str)   
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())

    # list of .tif files in the Input File Path     
    imageList = glob.glob('*.'+args['format'])
    # Create folders to store results if thay do no exist
    if not os.path.exists("BN"):
        os.makedirs("BN")
    
    for i in range(len(imageList)):
        name = os.path.basename(imageList[i])
        r = rasterio.open(imageList[i])            
        r2 = r.read()
        bn = np.apply_along_axis(BrigthnessNormalization, 0, r2)
        saveImage(bn, r)


