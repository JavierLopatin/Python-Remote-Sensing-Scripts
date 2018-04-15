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

from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin

######################################
# Data manipulation and transformation
######################################

##########

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
        import numpy as np
        # apply the normalization
        def norm(r):
            norm = r / np.sqrt( np.sum((r**2), 0) )
            return norm
        bn = np.apply_along_axis(norm, 2, X)
        return bn

##########

class MNF(BaseEstimator, TransformerMixin):
    """
    Apply a MNF transform to the image
    'img' must have (raw, column, band) shape
    """
    def __init__(self, n_components=1, BrightnessNormalization=False):
        self.n_components = n_components
        self.BrightnessNormalization = BrightnessNormalization
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        import numpy as np
        from sklearn.decomposition import PCA
        import pysptools.noise as ns

        X = X.astype('float32')
        # apply brightness normalization
        # if raster
        if self.BrightnessNormalization==True:
            def norm(r):
                norm = r / np.sqrt( np.sum((r**2), 0) )
                return norm
            if len(X.shape) == 3:
                X = np.apply_along_axis(norm, 2, X)
            # if 2D array
            if len(X.shape) == 2:
                X = np.apply_along_axis(norm, 0, X)
        w = ns.Whiten()
        wdata = w.apply(X)
        numBands = X.shape[2]
        h, w, numBands = wdata.shape
        X = np.reshape(wdata, (w*h, numBands))
        pca = PCA()
        mnf = pca.fit_transform(X)
        mnf = np.reshape(mnf, (h, w, numBands))
        mnf = mnf[:,:,:self.n_components]
        var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        return mnf, var

##########

def GLCM(outRaster, sizeWindow):
    """
    Run the GLCM textures and append them into one 3D array
    The "ndimage.generic_filter" funtion perform the moving window of size "window"
    """
    from skimage.feature import greycomatrix, greycoprops
    from scipy import ndimage
    from scipy.stats import entropy
    import numpy as np

    # prepare textures
    def homogeneity_fun(outRaster):
        """
        create Homogeneity using the GLCM function 
        of Skimage
        """
        if len(outRaster.shape) == 1:
            outRaster = np.reshape(outRaster, (-1, sizeWindow))
            
        glcm = greycomatrix(outRaster, [1], [0], symmetric = True, normed = True)
        return greycoprops(glcm, 'homogeneity')[0,0]
        
    def correlation_fun(outRaster):
        """
        create Correlation using the GLCM function 
        of Skimage
        """
        if len(outRaster.shape) == 1:
            outRaster = np.reshape(outRaster, (-1, sizeWindow))
            
        glcm = greycomatrix(outRaster, [1], [0], symmetric = True, normed = True)
        return greycoprops(glcm, 'correlation')[0,0]
    
    def contrast_fun(outRaster):
        """
        create contrast using the GLCM function 
        of Skimage
        """
        if len(outRaster.shape) == 1:
            outRaster = np.reshape(outRaster, (-1, sizeWindow))
            
        glcm = greycomatrix(outRaster, [1], [0], symmetric = True, normed = True)
        return greycoprops(glcm, 'contrast')[0,0]
     
    def  dissimilarity_fun(outRaster):
        """
        create dissimilarity_fun using the GLCM function 
        of Skimage
        """
        if len(outRaster.shape) == 1:
            outRaster = np.reshape(outRaster, (-1, sizeWindow))
            
        glcm = greycomatrix(outRaster, [1], [0], symmetric = True, normed = True)
        return greycoprops(glcm, 'dissimilarity')[0,0]

    # apply to moving window
    Variance      = ndimage.generic_filter(outRaster, np.var, size=sizeWindow)
    Contrast      = ndimage.generic_filter(outRaster, contrast_fun, size=sizeWindow)
    Dissimilarity = ndimage.generic_filter(outRaster, dissimilarity_fun, size=sizeWindow)
    Correlation   = ndimage.generic_filter(outRaster, correlation_fun, size=sizeWindow)
    Homogeneity   = ndimage.generic_filter(outRaster, homogeneity_fun, size=sizeWindow)
    Entropy       = ndimage.generic_filter(outRaster, entropy, size=sizeWindow)
    
    return np.dstack( (Variance, Contrast, Dissimilarity, Correlation, Homogeneity, Entropy) )

###########

def saveRaster(img, inputRaster, outputName):
    # Save created raster to TIFF
    # input img must be in (bands, row, column) shape
    import rasterio
    new_dataset = rasterio.open(outputName, 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=int(img.shape[0]), dtype=str(img.dtype),
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(img)
    new_dataset.close()

##########

def ExtractValues(raster, shp, func, ID):
    """
    Extract raster values by a shapefile mask.
    Several statistics are allowed:
    - min
    - max
    - mean [default]
    - count
    - sum
    - std
    - median
    - majority
    - minority
    - unique
    - range
    - nodata
    - percentile
    """
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterstats import zonal_stats
    import shapefile
    
    # Raster management
    r = rasterio.open(raster)
    affine = r.affine 
    array = r.read()
    bands = array.shape[0]
    bandNames = []
    for i in range(bands):
        a = "B" + str(i+1)
        bandNames.append(a)
    
    # Shapefile management
    shape = shapefile.Reader(shp)
    records = pd.DataFrame(shape.records())
    n = pd.DataFrame(shape.fields)[0].values.tolist().index(ID)
    id = records[n-1]

    # empty matrix to store results
    matrix = np.empty((len(records), bands+1), dtype=object)
    matrix[:,0] = id
    colnamesSHP = [ID]

    # Final colnames
    colNames = colnamesSHP + bandNames

    # Extract values
    for i in range(bands):
        # stats 
        array = r.read(i+1)
        stats = zonal_stats(shp, array, affine=affine, stats=func)
        x = pd.DataFrame(stats)
        matrix[:,i+1] = x[func]
    
    # set the final data frame
    df = pd.DataFrame(matrix, columns=colNames)
    return df

##########

def ExtractPointValues(raster, shp, ID):
    from rasterstats import point_query
    """ Extract raster values by a shapefile point mask.
    """
    import numpy as np
    import pandas as pd
    import rasterio
    import shapefile
    
    # Raster management
    r = rasterio.open(raster)
    affine = r.affine 
    array = r.read()
    bands = array.shape[0]
    bandNames = []
    for i in range(bands):
        a = "B" + str(i+1)
        bandNames.append(a)
    
    # Shapefile management
    shape = shapefile.Reader(shp)
    records = pd.DataFrame(shape.records())
    n = pd.DataFrame(shape.fields)[0].values.tolist().index(ID)
    id = records[n-1]

    # empty matrix to store results
    matrix = np.empty((len(records), bands+1), dtype=object)
    matrix[:,0] = id
    colnamesSHP = [ID]

    # Final colnames
    colNames = colnamesSHP + bandNames

    # Extract values
    for i in range(bands):
        # stats 
        array = r.read(i+1)
        stats = point_query(shp, array, affine=affine)
        x = pd.DataFrame(stats)
        matrix[:,i+1] = x[0]
    
    # set the final data frame
    df = pd.DataFrame(matrix, columns=colNames)
    return df

##########
