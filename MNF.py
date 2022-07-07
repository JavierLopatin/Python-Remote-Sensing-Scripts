#!/usr/bin/python
# -*- coding: latin-1 -*-

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
# Version: 2.0
# Last checked: 23/11/2020
#
# Usage:
#
# python MNF.py -i <Imput raster>
#               -c <Number of components [default = inputRaster bands]>
#               -p <Preprocessing: Brightness Normalization of Hyperspectral data [Optional]>
#               
# # --preprop [-p]: Brightness Normalization presented in Feilhauer et al., 2010
#
# # examples:
#             # Get the regular MNF transformation
#             python MNF.py -i raster.tif
#
#             # Get the regular MNF transformation of the first component
#             python MNF.py -i raster.tif -c 1
#
#             # with Brightness Normalization
#             python MNF_cmd.py -i raster.tif -p
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
from sklearn.decomposition import IncrementalPCA
import rasterio

try:
   import pysptools.noise as ns
except ImportError:
   print("ERROR: Could not import Pysptools Python library.")
   print("Check if Pysptools is installed.")

#%%
################
### Functions
################
def _norm(X):
    return X / np.sqrt( np.sum((X**2), 0) )
     
def brightNorm(X):  
    return np.apply_along_axis(_norm, 0, X)


#%%
### Run process

if __name__ == "__main__":

    # create the arguments for the algorithm
    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--inputRaster',
      help='Input raster', type=str, required=True)
    parser.add_argument('-c','--components',
      help='Number of components.', type=int, default=10000)
    parser.add_argument('-p','--preprop',
      help='Preprocessing: Brightness Normalization of Hyperspectral data [Optional].',
      action="store_true", default=False)
  
    parser.add_argument('--version', action='version', version='%(prog)s 2.0')
    args = vars(parser.parse_args())

    # data imputps/outputs
    #inRaster = 'forestSample3.tif'
    inRaster = args["inputRaster"]
    outMNF = inRaster[:-4] + "_MNF.tif"

    # Normalization?
    BrightnessNormalization = args["preprop"]
    
    # load raster
    with rasterio.open(inRaster) as r:
        meta = r.profile # metadata
        img = r.read()   # read as numpy array 
        count = r.count  # number of bands
        width = r.width
        height = r.height
        
    # set number of components to retrive
    if args['components'] == 10000:
         n_components = count
    else:
        n_components = args['components']
    
    # apply Brigthness Normalization if needed
    if BrightnessNormalization==True:
        print('Applying preprocessing...')
        img = brightNorm(img)
        outMNF = outMNF[:-4]+'_prepro.tif'
        print('Done!')
    #%%   
    # Apply NMF
    img = np.transpose(img, [1,2,0]) 
    img = img.reshape((width*height, count))
    
    # check for nan and inf
    if np.any(np.isinf(img))==True:
        img[img == np.inf] = 0
    if np.any(np.isnan(img))==True:
        img[img == np.nan] = 0
    
    # data whitenen
    img=ns.whiten(img)        
 
    # PCA
    print('Applying NMF transformation...')
    pca = IncrementalPCA()
    img = pca.fit_transform(img)
    print('Done!')
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    print("The explained variance per component is:")
    print(pca.explained_variance_ratio_)
    print("The accumulative explained variance per component is:")
    print(var)
    
    # save
    np.savetxt(outMNF[:-4]+'_variance.txt', pca.explained_variance_ratio_)
    np.savetxt(outMNF[:-4]+'_accVariance.txt', var)
    img = img.reshape((height, width, count))
    img = np.transpose(img, [2,0,1])    
    meta.update(count=n_components, dtype='float32')
    with rasterio.open(outMNF, "w", **meta) as dst:
        dst.write(img[:n_components, :, :]) 
        
