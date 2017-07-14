#! /usr/bin/env python

"""
Python script to perform and rastirize the Canupo funtion over LiDAR data
and rasterize outputs

Usage:

python canupo.py -i inData -s scales -r outResolution

    -- inData [-i]: text with the xyz LiDAR data
    -- scales [-s]: enter the start, end, and step for the scales
    -- resolution [-r]: resolution to export the rasters

example:
    python canupo.py -i lidar.txt -s 1 5 1 -r 1 # scales: 1,2,3,4,5 m; output resolution 1 m

Dependencies:
    - check the python libraries
    - GDAL
    - canupo.exe, msc_tool.exe and gdal_merge.py must be in the working directory
    - check directoy to the LasTools (i.e. to lasgrid.exe)

Author: Javier Lopatin
Email: javierlopatin@gmail.com
Last changes: 14/7/2017
Version: 1.0

Biliography:

Brodu, N. and Lague, D. (2012). 3D terrestrial lidar data classification of 
complex natural scenes using a multi-scale dimensionality criterion: 
Applications in geomorphology. ISPRS Journal of Photogrammetry and Remote
Sensing, vol. 68, p.121-134.
"""

import os, argparse, glob, shutil
import numpy as np
from subprocess import call

############################################
### Check dependecies folder directories ###
############################################

gdalDir = "C:/OSGeo4W64/bin/"
lastoolsDir = "C:/lastools/bin/"

#################
### Functions ###
#################

def RunCanupo(inData, scales, step):
    """
    Run Canupo function for four non-systematic scales.
    Then, transform the msc outputs to txt
    """

    # create temporal folder
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    
    # scales
    i0 = float(scales[0])
    i2 = float(scales[1])
    dif = float(scales[2])
    
    # run canupo
    outName = "tmp/out.msc"
    process = "canupo "+str(i0)+":"+str(dif)+":"+str(i2)+" : "+inData+" "+inData+" "+outName
    call(process)

    # msc2txt   
    process = "msc_tool xyz "+outName+" : "+outName[:-4]+".txt"
    call(process)

    """
    Reorder the outputs, separate the components, and rasterize them
    Then, make a raster stack with the outputs
    """
    # variables to use
    components = np.arange(i0, i2+(dif), dif)
    N = len(components)
    nonUsed = 3 + N*3
    colList = range(nonUsed, nonUsed+N,1)
    
    # load original coordinated
    df = np.loadtxt("tmp/out.txt", usecols=[0,1]) 
    # load components   
    df2 = np.loadtxt("tmp/out.txt", usecols=colList) 
    # merge         
    df3 = np.append(df, df2, axis=1)
   
    # loop through components
    for i in range(N):    
        # export results
        out = df3[:, (0,1,i+2)]
        outName = "tmp/"+inData[:-4]+"_comp_"+str(i+1)+".txt"
        np.savetxt(outName, out)
        # rasterize with lasgrid
        process = lastoolsDir+"lasgrid.exe -i "+outName+" -o "+outName[:-4]+".tif -step "+str(step)+" -elevation -average"
        call(process)
        
    # stack bands
    outName = inData[:-4]+"_"+str(i0)+"_"+str(i2)+".tif"
    tif_list = glob.glob("tmp/*.tif")
    tif_list = " ".join(tif_list)
    process = "python "+gdalDir+"gdal_merge.py -o "+outName+" "+tif_list+" -separate"
    call(process)
    
    # return information of the created raster
    call("gdalinfo " + outName)    
    
    # delate tables from memory
    del df, df2, df3
    
    # erase temporal folder
    shutil.rmtree("tmp")

          
##################
### Run script ###
##################

if __name__ == "__main__":
    
    # create the arguments for the algorithm
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inData', help='Input LiDAR table with xyz colums')   
    parser.add_argument('-s', '--scales', help='Input start, end, and step for the scales', nargs='+', type=float)  
    parser.add_argument('-r', '--resolution', help='Output raster resolution', default=1, type=float)  
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())
    
    inData = args["inData"]
    scales = args["scales"]
    step = args["resolution"]
        
    # execute script
    RunCanupo(inData, scales, step)
     
