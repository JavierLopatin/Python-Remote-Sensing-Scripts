#! /usr/bin/env python

########################################################################################################
#
# BrightnessNormalization.py
#
# A python script to perform Brigtness Normalization of hyperspectral data
#
# Info: The script apply the Brightness Normalization presented in
#       Feilhauer et al., 2010 to all rasters contained in a folder
#       with parallel processing of raster chunks. The whole raster image is never
#       completely loaded into memory
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Last changes: 19/05/2023
# Version: 2.0
#
# example: python BrightnessNormalization.py -i raster.tif
#
# Bibliography:
#
# Feilhauer, H., Asner, G. P., Martin, R. E., Schmidtlein, S. (2010): Brightness-normalized Partial Least Squares
# Regression for hyperspectral data. Journal of Quantitative Spectroscopy and Radiative Transfer 111(12-13),
# pp. 1947–1957. 10.1016/j.jqsrt.2010.03.007
#
########################################################################################################

from __future__ import division
import concurrent.futures
from functools import partial
import warnings
import os
import argparse
import numpy as np
import rasterio
from tqdm import tqdm

############
## Functions
############

def _normalize_vector(X):
    X = np.array(X, dtype='float32') # esto es lo nuevo
    return X / np.sqrt(np.sum((X**2)))

def _brightNorm(X):
    return np.apply_along_axis(_normalize_vector, 0, X).astype(np.float32)

def _parallel_process(inData, outData, do_work, count, n_jobs, chuckSize,
                      bandNames):
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
                    # save band description to metadata
                    for i in range(profile['count']):
                        dst.set_band_description(i + 1, bandNames[i])
    else:
        print('ERROR! chuckSize needs to be divisible by 16')


def brightNorm(inData, n_jobs=4, chuckSize=256):
    """
    Process the Brightness Normalization in parallel

    """

    # get names for output bands
    with rasterio.open(inData) as r:
        count = r.count      # number of bands

    bandNames = []
    for i in range(count):
        bandNames.append('B' + str([i]))

    # function to apply
    do_work = partial(_brightNorm)

    # out data
    outData = inData[:-4]+"_BN.tif"

    # apply parallel processing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _parallel_process(inData, outData, do_work, count, n_jobs, chuckSize,
                      bandNames)
    except AttributeError:
        print('ERROR in parallel processing...')

#%%
### Run process

if __name__ == "__main__":

   # create the arguments for the algorithm
   parser = argparse.ArgumentParser()

   parser.add_argument('-i','--input', help='Imput raster', type=str, required=True)
   parser.add_argument('--version', action='version', version='%(prog)s 2.0')
   args = vars(parser.parse_args())

   # input raster
   inData = args["input"]

   # Apply normalization
   brightNorm(inData)
