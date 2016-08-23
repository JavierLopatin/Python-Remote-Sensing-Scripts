#! /usr/bin/env python

########################################################################################################################
#
# MNF_cmd.py
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
# python MNF.py -f <Imput raster format [default = tif]> 
#               -c <Number of components> 
#               -m <Method option [default = 1]> 
#               -p <Preprocessing: Brightness Normalization of Hyperspectral data [Optional]> 
#               -s <Apply Savitzky Golay filtering [Optional]>
#               -v <Accumulated explained variance [optional]> 
#
# --format [-f]: Input raster formats (default tif)
# 
# -- method [-m]: Method options: 1 (default) regular MNF transformation.
#                                 2  MNF inverse transformation.
#
# --preprop [-p]: Brightness Normalization presented in Feilhauer et al., 2010
#
# --SavitzkyGolay [-s]: Apply Savitzky Golay filtering
#
# --variance [-v]: Get the accumulative explained variance of MNF components
#
# examples:   
#             # Get the accumulated explained variance
#             python MNF_cmd.py -c 1 -v
#
#             # with Brightness Normalization
#             python MNF_cmd.py -c 1 -v -p
#
#             # Get the regular MNF transformation
#             python MNF_cmd.py -f img -c 10 
#             python MNF_cmd.py -f img -c 10 -s # with Savitzky Golay
#
#             # with Brightness Normalization
#             python MNF_cmd.py -f tif -c 10 -p
#
#             # Get the reduced nose MNF with inverse transformation
#             python MNF_cmd.py -c 10 -m 2
#             python MNF_cmd.py -c 10 -m 2 -s # with Savitzky Golay
#
#             # with Brightness Normalization
#             python MNF_cmd.py -c 10 -m 2 -p
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
import os, glob, argparse
import numpy as np
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

## Functions 

def BrigthnessNormalization(img):
    r = img / np.sqrt( np.sum((img**2), 0) )
    return r

def MNF(img, n_components):
    mnf = ns.MNF()
    mnf.apply(img)
    if args["SavitzkyGolay"]==True:
        dn = ns.SavitzkyGolay()
        mnf[:,:,1:2] = dn.denoise_bands(mnf[:,:,1:2], 15, 2)
    r = mnf.get_components(n_components)
    return r

def MNF_reduce_component_2_noise_and_invert(img, n_components):
    # Reduce the second component noise and
    # return the inverse transform
    mnf = ns.MNF()
    tdata = mnf.apply(img)
    if args["SavitzkyGolay"]==True:
        dn = ns.SavitzkyGolay()
        tdata[:,:,1:2] = dn.denoise_bands(tdata[:,:,1:2], 15, 2)
    r = mnf.inverse_transform(tdata)
    r2 = r[:,:,1:n_components+1]
    return r2
    
def explained_variance(img):
    from sklearn.decomposition import PCA
    w = ns.Whiten()
    wdata = w.apply(img)
    numBands = r.count
    h, w, numBands = wdata.shape
    X = np.reshape(wdata, (w*h, numBands))
    pca = PCA()
    mnf = pca.fit_transform(X)
    return print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)) 

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
    im = np.ma.transpose(arr, [1,2,0])
    return im

def reshape_as_raster(arr):
    """Returns the array in a raster order
    by swapping the axes order from (rows, columns, bands)
    to (bands, rows, columns)
    Parameters
    ----------
    arr : array-like in the image form of (rows, columns, bands)
    """
    # swap the axes order from (rows, columns, bands) to (bands, rows, columns)
    im = np.transpose(arr, [2,0,1])
    return im

def saveMNF(img, inputRaster):
    # Save TIF image to a nre directory of name MNF
    img2 = reshape_as_raster(img)
    if args["preprop"]==True:
    	output = "BN_MNF/" + name[:-4] + "_BN_MNF.tif"
    else:
	output = "MNF/" + name[:-4] + "_MNF.tif"
    new_dataset = rasterio.open(output, 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=int(n_components), dtype=str(img.dtype),
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(img2)
    new_dataset.close()

### Run process
        
if __name__ == "__main__":
    # create the arguments for the algorithm
    parser = argparse.ArgumentParser()

    parser.add_argument('-f','--format', 
      help='Input raster format [default = tif].', type=str, default="tif")
    parser.add_argument('-c','--components', 
      help='Number of components.', type=int, required=True)
    parser.add_argument('-m','--method', 
      help='MNF method to apply: 1 (default) = regular MNF transformation; 2 = MNF invers transformation.', 
      type=int, default=1)
    parser.add_argument('-p','--preprop', 
      help='Preprocessing: Brightness Normalization of Hyperspectral data [Optional].', 
      action="store_true", default=False)
    parser.add_argument('-s','--SavitzkyGolay', 
      help='Apply Savitzky Golay filtering [Optional].',  action="store_true", default=False)
    parser.add_argument('-v','--variance', 
      help='Accumulated explained variance.', action="store_true", default=False)
    
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())
        
    # Define number of components for the MNF
    n_components = args['components']
    # list of .tif files in the Input File Path     
    imageList = glob.glob('*.'+args['format'])
    # Create folders to store results if thay do no exist
    if not os.path.exists("MNF"):
        os.makedirs("MNF")

    if args['variance']==True:
        # Show the accumulated explained variance
        for i in range(len(imageList)):
            name = os.path.basename(imageList[i])
            r = rasterio.open(imageList[i])            
            r2 = r.read()
            # Apply Brightness Normalization if the option -p is added
            if args["preprop"]==True:
                     r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
            img = reshape_as_image(r2)
            print("Accumulated explained variances of " + name + "are:")
            explained_variance(img)
    else:  
        if args['method']==1:
            for i in range(len(imageList)):
                # Load raster/convert to ndarray format
                name = os.path.basename(imageList[i])
                r = rasterio.open(imageList[i])            
                r2 = r.read()
                # Apply Brightness Normalization if the option -p is added
                if args["preprop"]==True:
                     r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
                img = reshape_as_image(r2)
                # Apply MNF -m 1
                print("Creating MNF components of " + name)
                mnf = MNF(img, n_components)
                saveMNF(mnf, r)
                    
        elif args['method']==2:
            for i in range(len(imageList)):
                # Load raster/convert to ndarray format
                name = os.path.basename(imageList[i])
                r = rasterio.open(imageList[i])            
                r2 = r.read()
                # Apply Brightness Normalization if the option -p is added
                if args["preprop"]==True:
                     r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
                img = reshape_as_image(r2)
                # Apply MNF -m 2
                print("Creating MNF components of " + name)
                mnf = MNF_reduce_component_2_noise_and_invert(img, n_components)
                saveMNF(mnf, r) 
        else: 
            print('ERROR!. Command should have the form:')
            print('python MNF.py -f <Imput raster formar> -c <Number of components> -m <Method option>[optional] -v <Accumulated explained variance>[optional]')
            print("")
            print("Method options: 1 (default) regular MNF transformation")
            print("                2  inverse transform")
            print("")
            print("-p or --preprop: Apply Broghtness Normalization of hyperspectral data")
            print("")
            print("-s or --Savitzky Golay: Use Savitzky Golay methods")
            print("")
            print("example: python MNF_cmd.py -f tif -c 10")
