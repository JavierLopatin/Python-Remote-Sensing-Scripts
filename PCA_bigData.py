#!/usr/bin/python
# -*- coding: utf-8 -*-
########################################################################################################################
#
# PCA_bigData.py
# A python script to perform PCA transformation to remote sesning data without loading the entire data
# into local memory. The script uses parallel processing.
#
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Date: 27/11/2020
# Version: 1.0
#
# Usage:
#
# python MNF.py -i <Imput raster>
#               -c <Number of components [default = inputRaster bands]>
#               -p <Preprocessing: Brightness Normalization of spectral data [Optional]>
#               -n <Raster chunk size to be loaded into memory. It must be divisible by 16>
#               -j <n_jobs; number of parallel works to use>
#               
# # --preprop [-p]: Brightness Normalization presented in Feilhauer et al., 2010
#
# # examples:
#             # Get the regular MNF transformation
#             python PCA_bigData.py -i raster.tif
#
#             # Get the regular MNF transformation of the first component
#             python PCA_bigData.py -i raster.tif -c 1
#
#             # with Brightness Normalization
#             python PCA_bigData.py -i raster.tif -p
#
#
#
# Bibliography:
#
# Feilhauer, H., Asner, G. P., Martin, R. E., Schmidtlein, S. (2010): Brightness-normalized Partial Least Squares
# Regression for hyperspectral data. Journal of Quantitative Spectroscopy and Radiative Transfer 111(12-13),
# pp. 1947-1957. 10.1016/j.jqsrt.2010.03.007
#
# C-I Change and Q Du. 1999. Interference and Noise-Adjusted Principal Components Analysis.
# IEEE TGRS, Vol 36, No 5.
#
########################################################################################################################



from __future__ import division
import concurrent.futures
from functools import partial
import warnings
import argparse
import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm 
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
    X = X.astype(np.float32)
    return X / np.sqrt( np.sum((X**2), 0) )
     
def _brightNorm(X):  
    return np.apply_along_axis(_norm, 0, X)

def _parallel_process(inData, outData, do_work, count, n_jobs, chuckSize):
    """
    Process infile block-by-block with parallel processing
    and write to a new file.
    chunckSize needs to be divisible by 16

    """
    if chuckSize % 16 == 0:
        # apply parallel processing with rasterio
        with rasterio.Env():
            with rasterio.open(inData) as src:
                # Create a destination dataset based on source params. The
                # destination will be tiled, and we'll process the tiles
                # concurrently.
                profile = src.profile
                profile.update(blockxsize=chuckSize, blockysize=chuckSize,
                               count=count, dtype='float32', tiled=True)

                with rasterio.open(outData, "w", **profile) as dst:
                    # Materialize a list of destination block windows
                    # that we will use in several statements below.
                    windows = [window for ij, window in dst.block_windows()]
                    # This generator comprehension gives us raster data
                    # arrays for each window. Later we will zip a mapping
                    # of it with the windows list to get (window, result)
                    # pairs.
                    data_gen = (src.read(window=window) for window in windows)
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=n_jobs
                    ) as executor:
                        # Map the a function over the raster
                        # data generator, zip the resulting iterator with
                        # the windows list, and as pairs come back we
                        # write data to the destination dataset.
                        for window, result in zip(
                            tqdm(windows), executor.map(do_work, data_gen)
                        ):
                            dst.write(result, window=window)
    else:
        print('ERROR! chuckSize needs to be divisible by 16')


def brightNorm(inData, n_jobs=4, chuckSize=256):
    """
    Process the Brightness Normalization in parallel

    """

    # get names for output bands
    with rasterio.open(inData) as r:
        count = r.count      # number of bands
 
      # call _getPheno2 function to lcoal
    do_work = partial(_brightNorm)
    
    # out data
    outData = inData[:-4]+"_BN.tif"

    # apply PhenoShape with parallel processing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _parallel_process(inData, outData, do_work, count, n_jobs, chuckSize)
    except AttributeError:
        print('ERROR in parallel processing...')

def PCA_pred(w):
    """
    Process the Brightness Normalization in parallel

    """
    count, width, height = w.shape
    w = np.transpose(w, [1,2,0]) 
    w = w.reshape((width*height, count))
    if np.any(np.isinf(w))==True:
        w[w == np.inf] = 0
    if np.any(np.isnan(w))==True:
        w[w == np.nan] = 0                
    # predict PCA
    w = ipca.transform(w)
    w = w.reshape((height, width, count))
    w = np.transpose(w, [2,0,1]) 
    w = w[:n_components,:,:]
    return w.astype(np.float32)
  

def PCA_train(inData, outData, n_jobs, chuckSize):
    """
    Process infile block-by-block with parallel processing
    and write to a new file.
    chunckSize needs to be divisible by 16

    """
    do_work = partial(PCA_pred) # pass prediction function to loop                      
    if chuckSize % 16 == 0:
          with rasterio.Env():
              with rasterio.open(inData) as src:
                  count = src.count
                  profile = src.profile
                  profile.update(blockxsize=chuckSize, blockysize=chuckSize,
                                 count=n_components, dtype='float32', tiled=True)
                  with rasterio.open(outData, "w", **profile) as dst:
                      windows = [window for ij, window in dst.block_windows()]
                      data_gen = (src.read(window=window) for window in windows) 
                      # train incrementalPCA with raster chunks
                      print('Training incrementalPCA with chuncks...')
                      for i in tqdm(range(len(windows))):
                          w = src.read(window=windows[i]) # read window-by-window
                          width = windows[i].width
                          height = windows[i].height
                          w = np.transpose(w, [1,2,0]) 
                          w = w.reshape((width*height, count))
                          # check for nan and inf
                          if np.any(np.isinf(w))==True:
                              w[w == np.inf] = 0
                          if np.any(np.isnan(w))==True:
                              w[w == np.nan] = 0
                          # train pca partially
                          ipca.partial_fit(w)
                      print('')
                      print("The explained variance per component is:")
                      print(ipca.explained_variance_ratio_)
                      print("The accumulative explained variance per component is:")
                      print(np.cumsum(np.round(ipca.explained_variance_ratio_, decimals=4)*100))
                      print('')
                      print('Applying PCA transformation and saving into disk...')  
                      with concurrent.futures.ProcessPoolExecutor(
                          max_workers=n_jobs
                      ) as executor:
                          for window, result in zip(  
                              tqdm(windows), executor.map(do_work, data_gen)
                          ):
                              dst.write(result, window=window)
    else:
        print('ERROR! chuckSize needs to be divisible by 16')

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
    parser.add_argument('-j','--n_jobs',
      help='Number of cores to be used in parallel processing.', 
      type=int, default=4)
    parser.add_argument('-n','--chuckSize',
      help='Size of the raster chunks to be loaded into memory [needs to be divisible by 16]', 
      type=int, default=256)
  
    parser.add_argument('--version', action='version', version='%(prog)s 2.0')
    args = vars(parser.parse_args())

    # data imputps/outputs
    # inData = 'trialdata.tif'
    inData = args["inputRaster"]
    n_jobs = args["n_jobs"]
    chuckSize = args["chuckSize"]
    
    # Normalization?
    BrightnessNormalization = args["preprop"]
    
       
    # set number of components to retrive
    with rasterio.open(inData) as r:
        count = r.count
    if args['components'] == 10000:
         n_components = count
    else:
        n_components = args['components']
    
    # apply Brigthness Normalization if needed
    if BrightnessNormalization==True:
        print('Applying preprocessing...')
        brightNorm(inData)
        print('')
    
    # PCA
    ipca = IncrementalPCA() 
    if BrightnessNormalization==True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PCA_train(inData[:-4]+"_BN.tif", inData[:-4]+"_prepro_PCA.tif",
                      n_jobs, chuckSize)
            # delete temporal preprocessing file
            os.remove(inData[:-4]+"_BN.tif")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PCA_train(inData, inData[:-4]+"_PCA.tif", n_jobs, chuckSize)
        
        