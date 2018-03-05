#! /usr/bin/env python

########################################################################################################################
#
# MNF.py
# A python script to perform MNF transformation to remote sesning data.
#
# Info: The script perform MNF transformation to all raster images stored in a folder. 
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Date: 09/08/2016
# Version: 1.0
#
# Usage:
#
# python MNF.py -i <Imput raster> 
#               -c <Number of components [default = inputRaster bands]> 
#               -p <Preprocessing: Brightness Normalization of Hyperspectral data [Optional]> 
#               -s <Apply Savitzky Golay filtering [Optional]>
#            
# # --preprop [-p]: Brightness Normalization presented in Feilhauer et al., 2010
#
# --SavitzkyGolay [-s]: Apply Savitzky Golay filtering
#
# # examples:   
#             # Get the regular MNF transformation
#             python MNF.py -i raster
#
#             # with Savitzky Golay 
#             python MNF.py -i raster -s 
#
#             # with Brightness Normalization
#             python MNF_cmd.py -i raster -p
#
# 
#
# Bibliography:
#
# Feilhauer, H., Asner, G. P., Martin, R. E., Schmidtlein, S. (2010): Brightness-normalized Partial Least Squares
# Regression for hyperspectral data. Journal of Quantitative Spectroscopy and Radiative Transfer 111(12-13),
# pp. 1947–1957. 10.1016/j.jqsrt.2010.03.007
#
# C-I Change and Q Du. 1999. Interference and Noise-Adjusted Principal Components Analysis. 
# IEEE TGRS, Vol 36, No 5.
#
########################################################################################################################

from __future__ import division
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
try:
   import rasterio
except ImportError:
   print("ERROR: Could not import Rasterio Python library.")
   print("Check if Rasterio is installed.")

try:
   import pysptools.noise as ns
except ImportError:
   print("ERROR: Could not import Pysptools Python library.")
   print("Check if Pysptools is installed.")

################
### Functions 
################

def BrigthnessNormalization(img):
    """
    Brightness normaliyation for hyperspectral data.
    See Feilhauer et al. (2010)
    """ 
    r = img / np.sqrt( np.sum((img**2), 0) )
    return r

def MNF(img):
    """
    Apply a MNF transform to the image
    'img' must have (raw, column, band) shape
    """
    w = ns.Whiten()
    wdata = w.apply(img)
    numBands = img.shape[2]
    h, w, numBands = wdata.shape
    X = np.reshape(wdata, (w*h, numBands))
    pca = PCA()
    mnf = pca.fit_transform(X)
    mnf = np.reshape(mnf, (h, w, numBands))
    if args["SavitzkyGolay"]==True:
        dn = ns.SavitzkyGolay()
        mnf[:,:,1:2] = dn.denoise_bands(mnf[:,:,1:2], 15, 2)
    if args["components"] == True:
        n_components = args['components']
    else:
        n_components = img.shape[2]
    r = mnf[:,:,:n_components]
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    return r, var

def saveMNF(img, inputRaster):
    # Save TIF image to a nre directory of name MNF
    img2 = np.transpose(img, [2,0,1]) # get to (band, raw, column) shape 
    output = outMNF 
    if args["preprop"]==True:
        output = output[:-4] + "_BN.tif"    
    if args["SavitzkyGolay"]==True:
        output = output[:-4] + "_Savitzky.tif"
    new_dataset = rasterio.open(output , 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=img.shape[2], dtype=str(img.dtype),
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(img2)
    new_dataset.close()

### Run process
        
if __name__ == "__main__":
    
    # create the arguments for the algorithm
    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--inputRaster', 
      help='Input raster', type=str, required=True)
    parser.add_argument('-c','--components', 
      help='Number of components.', type=int, default=False)
    parser.add_argument('-p','--preprop', 
      help='Preprocessing: Brightness Normalization of Hyperspectral data [Optional].', 
      action="store_true", default=False)
    parser.add_argument('-s','--SavitzkyGolay', 
      help='Apply Savitzky Golay filtering [Optional].',  action="store_true", default=False)
        
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())
     
    # data imputps/outputs
    inRaster = args["inputRaster"]
    outMNF = inRaster[:-4] + "_MNF.tif"
    
    # load raster
    r = rasterio.open(inRaster)            
    r2 = r.read() # transform to array
    
    # Apply Brightness Normalization if the option -p is added
    if args["preprop"]==True:
        r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
        r2 = np.nan_to_num(r2)
        
    img = np.transpose(r2, [1,2,0]) # get to (raw, column, band) shape 
    # Apply MNF
    print("Creating MNF components of " + inRaster)
    mnf, var = MNF(img)
    print("The accumulative explained variance per component is:")
    print(var)
   
    # save the MNF image and explained variance
    saveMNF(mnf, r) 
    bandNames = []
    for i in range(mnf.shape[2]):
        a = "MNF" + str(i+1)
        bandNames.append(a)
    bandNames = pd.DataFrame(bandNames)
    variance = pd.DataFrame(var)
    txtOut = pd.concat([bandNames, variance], axis=1)
    txtOut.columns=["Bands", "AccVariance"]
    txtOut.to_csv(outMNF[:-4] + ".csv", index=False, header=True, na_rep='NA') 
     
        
        