#! /usr/bin/env python

########################################################################################################
#
# BrightnessNormalization.py
#
# A python script to perform Brigtness Normalization of hyperspectral data
#
# Info: The script apply the Brightness Normalization presented in
#       Feilhauer et al., 2010 to all rasters contained in a folder
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Last changes: 07/12/2016
# Version: 1.0
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
import os, argparse
import numpy as np
try:
   import rasterio
except ImportError:
   print("ERROR: Could not import Rasterio Python library.")
   print("Check if Rasterio is installed.")

############
## Functions
############

class BrigthnessNormalization(BaseEstimator, TransformerMixin):
    """
    Brightness transformation of spectra as described in
    Feilhauer, H., Asner, G. P., Martin, R. E., Schmidtlein, S. (2010): 
    Brightness-normalized Partial Least Squares Regression for hyperspectral data. 
    Journal of Quantitative Spectroscopy and Radiative Transfer 111(12-13),
    1947–1957. 10.1016/j.jqsrt.2010.03.007
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    
    def __init__(self, img = True):
        self.img = img
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        def norm(r):
            norm = r / np.sqrt( np.sum((r**2), 0) )
            return norm
        bn = np.apply_along_axis(norm, 2, X)
        return bn

def saveRaster(img, inputRaster, outputName):
    # Save created raster to TIFF
    new_dataset = rasterio.open(outputName, 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=int(img.shape[0]), dtype=str(img.dtype),
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(img)
    new_dataset.close()

### Run process

if __name__ == "__main__":

   # create the arguments for the algorithm
   parser = argparse.ArgumentParser()

   parser.add_argument('-i','--input', help='Imput raster', type=str, required=True)
   parser.add_argument('--version', action='version', version='%(prog)s 1.0')
   args = vars(parser.parse_args())

   # input raster
   image = args["input"]

   name = os.path.basename(image)
   r = rasterio.open(image)
   img = r.read().astype('float32')
   img = np.transpose(img, [1,2,0]) # get to (raw, column, band) shape 
   # apply normalization
   bn = BrigthnessNormalization()
   bn = bn.fit_transform(img)
   bn = np.transpose(bn, [2,0,1]) # get to (raw, column, band) shape 
   # save created raster
   saveRaster(bn, r, name[:-4]+"_BN.tif")
