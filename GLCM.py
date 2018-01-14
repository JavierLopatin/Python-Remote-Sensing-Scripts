# -*- coding: utf-8 -*-
"""
Python script to GLCM texture indices

Usage:
python GLCM.py -i inData -w WindowSize -p PreProcessing
    -- inData [-i]: text with input raster image
    -- WindowSize [-w]: Moving window size. It must be odd. Coud be one or several  values
    -- BandToUse [-b]: Band to use form the raster stack (default 1).
    -- RGB [-r]: Use intensitz transformation if inData is an RGB image
    
examples:
    - python GLCM.py -i raster.tif # use the first band and a 3 X 3 moving window
    - python GLCM.py -i raster.tif - w 7 # Use a 7 X 7 moving window
    - python GLCM.py -i raster.tif - w 7 -b 3 # Use a 7 X 7 moving window and the third band
    - python GLCM.py -i raster.tif -r # Use the RGN intensitz transformation
    - python GLCM.py -i raster.tif -w 3 5 7 9 # use several window sizes to create several outputs


Author: Javier Lopatin
Email: javierlopatin@gmail.com
Last changes: 12/1/2018
Version: 1.0

"""

import argparse, os
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage
from scipy.stats import entropy
from scipy.misc import imresize
try:
   import rasterio
except ImportError:
   print("ERROR: Could not import Rasterio Python library.")
   print("Check if Rasterio is installed.")

################
#### Functions
################

def read_raster(inputRaster):
    """    
    Read image into Numpy Array
    """
    img = rasterio.open(inputRaster)
    raster = img.read()
   
   # transform to RGB intensity
    if args["RGB"] == True:  
        # RGB bands
        r = raster[0,:,:]
        g = raster[1,:,:]
        b = raster[2,:,:]
        # Transform RGB to intensity (or lightness) of the HSL color scales
        # Preserves distances and angles from the geometry of the RGB cube
        outRaster = imresize( (0.2989 * r) + (0.5870 * g) + (0.1140 * b), 100 )
    
    else: # otherwise select a band from a raster stack
        outRaster = imresize(raster[band-1,:,:], 100) 
               
    return img, outRaster 
    
    
def save_raster(array, inputRaster, size):
    """    
    Save TIF image to a new directory
    """    
    # create temporal folder
    if not os.path.exists("GLCM"):
        os.makedirs("GLCM")
        
    # reshape array
    array = reshape_as_raster(array)
    output = "GLCM/" + inputRaster.name[:-4] + "_GLCM_size_" + str(size) + ".tif"
    new_dataset = rasterio.open(output, 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=array.shape[0], dtype=str(array.dtype),
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(array)
    new_dataset.close()
  
def reshape_as_image(arr):
    """Returns the source array reshaped into the order
    expected by image processing and visualization software
    (matplotlib, scikit-image, etc)
    by swapping the axes order from (bands, rows, columns)
    to (rows, columns, bands)
    Parameters
    ----------
    source : array-like in a of format (bands, rows, columns)
    """
    # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
    return np.transpose(arr, [1,2,0])
        
def reshape_as_raster(array):
    """Returns the array in a raster order
    by swapping the axes order from (rows, columns, bands)
    to (bands, rows, columns)
    Parameters
    ----------
    arr : array-like in the image form of (rows, columns, bands)
    """
    # swap the axes order from (rows, columns, bands) to (bands, rows, columns)
    return np.transpose(array, [2,0,1])

def homogeneity_fun(outRaster):
    """
    create Homogeneity using the GLCM function 
    of Skimage
    """
    if len(outRaster.shape) == 1:
        outRaster = np.reshape(outRaster, (-1, sizeWindow))
        
    glcm = greycomatrix(outRaster, [1], [0],  symmetric = True, normed = True)
    return greycoprops(glcm, 'homogeneity').sum()
    
def correlation_fun(outRaster):
    """
    create Correlation using the GLCM function 
    of Skimage
    """
    if len(outRaster.shape) == 1:
        outRaster = np.reshape(outRaster, (-1, sizeWindow))
        
    glcm = greycomatrix(outRaster, [1], [0],  symmetric = True, normed = True)
    return greycoprops(glcm, 'correlation').sum()

def contrast_fun(outRaster):
    """
    create contrast using the GLCM function 
    of Skimage
    """
    if len(outRaster.shape) == 1:
        outRaster = np.reshape(outRaster, (-1, sizeWindow))
        
    glcm = greycomatrix(outRaster, [1], [0],  symmetric = True, normed = True)
    return greycoprops(glcm, 'contrast').sum()
 
def  dissimilarity_fun(outRaster):
    """
    create dissimilarity_fun using the GLCM function 
    of Skimage
    """
    if len(outRaster.shape) == 1:
        outRaster = np.reshape(outRaster, (-1, sizeWindow))
        
    glcm = greycomatrix(outRaster, [1], [0],  symmetric = True, normed = True)
    return greycoprops(glcm, 'dissimilarity').sum()
      
def run_textures(outRaster, sizeWindow):
    """
    Run the GLCM textures and append them into one
    3D array
    The "ndimage.generic_filter" funtion perform the moving window of size "window"
    """
    Variance      = ndimage.generic_filter(outRaster, np.var, size=sizeWindow)
    Contrast      = ndimage.generic_filter(outRaster, contrast_fun, size=sizeWindow)
    Dissimilarity = ndimage.generic_filter(outRaster, dissimilarity_fun, size=sizeWindow)
    Correlation   = ndimage.generic_filter(outRaster, correlation_fun, size=sizeWindow)
    Homogeneity   = ndimage.generic_filter(outRaster, homogeneity_fun, size=sizeWindow)
    Entropy       = ndimage.generic_filter(outRaster, entropy, size=sizeWindow)  
    
    return np.dstack( (Variance, Contrast, Dissimilarity, Correlation, Homogeneity, Entropy) )


###############
### Run script
###############

if __name__ == "__main__":
    
    # create the arguments to call from terminal
    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--inputRaster', 
      help='Input raster', type=str, required=True)
    parser.add_argument('-w','--WindowSize', 
      help='One or a list of moving window sizes (default 3).', nargs='+', type=int, default=3)
    parser.add_argument('-b','--BandToUse', 
      help='Band to use for the textures (default 1).', type=int, default=1)
    parser.add_argument('-r','--RGB', 
      help='Transform RGB image to intensity.', action="store_true", default=False)
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())
     
    # input variables:
    inRaster = args["inputRaster"]
    window   = args["WindowSize"]
    band = args["BandToUse"]
    
    # load image and transfor to intensity    
    img, raster = read_raster(inRaster)
    
    # perform GLCM textures:
    # If window size is a list
    if isinstance(window, list):
       for i in range(len(window)):
           sizeWindow = window[i]
           GLCM = run_textures(raster, sizeWindow)
           save_raster(GLCM, img, sizeWindow)
    
    else: # otherwise is only one size
        sizeWindow = window
        GLCM = run_textures(raster, window)
        save_raster(GLCM, img, window)

