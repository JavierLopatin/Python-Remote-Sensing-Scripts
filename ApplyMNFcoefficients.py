#! /usr/bin/env python

########################################################################################################################
#
# ApplyMNFcoefficients.py
# A python script to perform MNF transformation to one image and then apply the coefficients to other images.
#
# WARNING!: this assume that all images are in the same format
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
# python MNF.py -i <Input raster from which copy the MNF coefficients> 
#               -c <Number of components> 
#               -m <Method option [default = 1]> 
#               -p <Preprocessing: Brightness Normalization of Hyperspectral data [Optional]> 
#               -s <Apply Savitzky Golay filtering [Optional]>
#               -v <Accumulated explained variance [Optional]> 
#
# --inputImage [-i]: input raster from which copy the MNF coefficients
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
#             python ApplyMNFcoefficients.py -i image.tif -c 1 -v
#
#             # with Brightness Normalization
#             python ApplyMNFcoefficients.py -i image.tif-c 1 -v -p
#
#             # Get the regular MNF transformation
#             python ApplyMNFcoefficients.py -i image.tif -c 10 
#             python ApplyMNFcoefficients.py -i image.tif -c 10 -s # with Savitzky Golay
#
#             # with Brightness Normalization
#             python ApplyMNFcoefficients.py -i image.tif -c 10 -p
#
#             # Get the reduced nose MNF with inverse transformation
#             python ApplyMNFcoefficients.py -i image.tif-c 10 -m 2
#             python ApplyMNFcoefficients.py -i image.tif-c 10 -m 2 -s # with Savitzky Golay
#
#             # with Brightness Normalization
#             python ApplyMNFcoefficients.py -i image.tif-c 10 -m 2 -p
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


import os, glob, argparse
import numpy as np
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

## Functions 

def BrigthnessNormalization(img):
    r = img / np.sqrt( np.sum((img**2), 0) )
    return r

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

###############################

if __name__ == "__main__":

 # create the arguments for the algorithm
    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--inputImage', 
      help='Input raster from which copy the MNF coefficients', type=str)
    parser.add_argument('-c','--components', 
      help='Number of components', type=int, required=True)
    parser.add_argument('-m','--method', 
      help='MNF method to apply: 1 (default) = regular MNF transformation; 2 = MNF invers transformation', 
      type=int, default=1)
    parser.add_argument('-p','--preprop', 
      help='Preprocessing: Brightness Normalization of Hyperspectral data [Optional]',  
      action="store_true", default=False)
    parser.add_argument('-s','--SavitzkyGolay', 
      help='Apply Savitzky Golay filtering [Optional]',  action="store_true", default=False)
    parser.add_argument('-v','--variance', 
      help='Accumulated explained variance', action="store_true", default=False)   
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    
    args = vars(parser.parse_args())
               
    # Define number of components for the MNF
    n_components = args['components'] 
    # Input image
    inImage = args['inputImage']
    # list of .tif files in the Input File Path     
    imageList = glob.glob('*.'+inImage[-3:])
    imageList.remove(inImage) # remove the Input image
    
    # Create folders to store results if thay do no exist
    if args["preprop"]==True:
    	if not os.path.exists("BN_MNF"):
        	os.makedirs("BN_MNF")
    else:
    	if not os.path.exists("MNF"):
        	os.makedirs("MNF")
        
    if args['variance']==True:
        # Show the accumulated explained variance
        name = os.path.basename(inImage)
        r = rasterio.open(inImage)            
        r2 = r.read()
        # Apply Brightness Normalization if the option -p is added
        if args["preprop"]==True:
            bn = np.apply_along_axis(BrigthnessNormalization, 0, r2)
            r2 = reshape_as_image(bn) 
        img = reshape_as_image(r2)
        print("Accumulated explained variances of " + name + "are:")
        explained_variance(img)
    
    else:
        if args['method']==1:
            # Load raster/convert to ndarray format
            name = os.path.basename(inImage)
            r = rasterio.open(inImage)            
            r2 = r.read()
            # Apply Brightness Normalization if the option -p is added
            if args["preprop"]==True:
                r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
            img = reshape_as_image(r2)
            # Apply MNF -m 1
            print("Creating MNF components of " + name)
            # Apply MNF namualy acording to pysptools
            w = ns.Whiten()
            wdata = w.apply(img)
            numBands = r.count
            h, w, numBands = wdata.shape
            X = np.reshape(wdata, (w*h, numBands))
            pca = PCA()
            mnf = pca.fit_transform(X)
            mnf = np.reshape(mnf, (h, w, numBands))
            if args["SavitzkyGolay"]==True:
                dn = ns.SavitzkyGolay()
                mnf[:,:,1:2] = dn.denoise_bands(mnf[:,:,1:2], 15, 2)
            mnf = mnf[:,:,:n_components]
            saveMNF(mnf, r)
            
            # Apply MNF coefficients to the other images
            for i in range(len(imageList)):
                name = os.path.basename(imageList[i])
                r = rasterio.open(imageList[i])            
                r2 = r.read()
                # Apply Brightness Normalization if the option -p is added
                if args["preprop"]==True:
                    r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
                img = reshape_as_image(r2)
                # Apply MNF -m 1
                print("Creating MNF components of " + name)
                # Apply MNF namualy acording to pysptools
                w = ns.Whiten()
                wdata = w.apply(img)
                numBands = r.count
                h, w, numBands = wdata.shape
                Y = np.reshape(wdata, (w*h, numBands))
                mnf = pca.fit_transform(Y)
                mnf = np.reshape(mnf, (h, w, numBands))
                if args["SavitzkyGolay"]==True:
                    dn = ns.SavitzkyGolay()
                    mnf[:,:,1:2] = dn.denoise_bands(mnf[:,:,1:2], 15, 2)
                mnf = mnf[:,:,:n_components]
                saveMNF(mnf, r)
                
        elif args['method']==2:
            # Load raster/convert to ndarray format
            name = os.path.basename(inImage)
            r = rasterio.open(inImage)            
            r2 = r.read()
            # Apply Brightness Normalization if the option -p is added
            if args["preprop"]==True:
                r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
            img = reshape_as_image(r2)
            # Apply MNF -m 1
            print("Creating MNF components of " + name)
            # Apply MNF namualy acording to pysptools
            w = ns.Whiten()
            wdata = w.apply(img)
            numBands = r.count
            h, w, numBands = wdata.shape
            X = np.reshape(wdata, (w*h, numBands))
            pca = PCA()
            mnf = pca.fit_transform(X)
            mnf = np.reshape(mnf, (h, w, numBands))
            if args["SavitzkyGolay"]==True:
                dn = ns.SavitzkyGolay()
                mnf[:,:,1:2] = dn.denoise_bands(mnf[:,:,1:2], 15, 2)
            a = pca.inverse_transform(mnf)
            mnf = a[:,:,1:n_components+1]
            saveMNF(mnf, r)
            
            # Apply MNF coefficients to the other images
            for i in range(len(imageList)):
                name = os.path.basename(imageList[i])
                r = rasterio.open(imageList[i])            
                r2 = r.read()
                # Apply Brightness Normalization if the option -p is added
                if args["preprop"]==True:
                    r2 = np.apply_along_axis(BrigthnessNormalization, 0, r2)
                img = reshape_as_image(r2)
                # Apply MNF -m 1
                print("Creating MNF components of " + name)
                # Apply MNF namualy acording to pysptools
                w = ns.Whiten()
                wdata = w.apply(img)
                numBands = r.count
                h, w, numBands = wdata.shape
                Y = np.reshape(wdata, (w*h, numBands))
                mnf = pca.fit_transform(Y)
                mnf = np.reshape(mnf, (h, w, numBands))
                if args["SavitzkyGolay"]==True:
                    dn = ns.SavitzkyGolay()
                    mnf[:,:,1:2] = dn.denoise_bands(mnf[:,:,1:2], 15, 2)
                a = pca.inverse_transform(mnf)
                mnf = a[:,:,1:n_components+1]
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
