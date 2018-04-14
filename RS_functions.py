# -*- coding: utf-8 -*-

########################################################################################################
#
# RS_functions.py
#
# A set of functions for remote sensing tasks
#
# Info: The script apply the Brightness Normalization presented in
#       Feilhauer et al., 2010 to all rasters contained in a folder
#
# Author: Javier Lopatin
# Email: javierlopatin@gmail.com
# Last changes: 14/04/2018
# Version: 1.0
#
########################################################################################################

from sklearn.base import BaseEstimator, TransformerMixin

######################################
# Data manipulation and transformation
######################################

class BrigthnessNormalization(BaseEstimator, TransformerMixin):
    """
    Brightness transformation of spectra as described in
    Feilhauer, H., Asner, G. P., Martin, R. E., Schmidtlein, S. (2010): 
    Brightness-normalized Partial Least Squares Regression for hyperspectral data. 
    Journal of Quantitative Spectroscopy and Radiative Transfer 111(12-13),
    1947–1957. 10.1016/j.jqsrt.2010.03.007
    """
    def __init__(self, img = True):
        self.img = img
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        # apply the normalization
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
