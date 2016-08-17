

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

def MNF(img, n_components):
    mnf = ns.MNF()
    mnf.apply(img)
    r = mnf.get_components(n_components)
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
    return print(pca.explained_variance_ratio_.cumsum()) 

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
    output = "MNF/" + name[:-4] + "_MNF.tif"
    new_dataset = rasterio.open(output, 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=int(n_components), dtype=inputRaster.dtypes[0],
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(img2)
    new_dataset.close()

###############################

#if __name__ == "__main__":
            
# Define number of components for the MNF
n_components = args['method']

# list of .tif files in the Input File Path     
#imageList = glob.glob('*.'+args['format'])

# Create folders to store results if thay do no exist
if not os.path.exists("MNF"):
    os.makedirs("MNF")
    
### one image
inImage = "1907_potVal_sp_6_7.dat"
outImage = "1907_potVal_sp_6_7.tif"

# Load raster/convert to ndarray format
name = os.path.basename(inImage)
r = rasterio.open(inImage)            
r2 = r.read()
# Apply Brightness Normalization if the option -p is added
if args["preprop"]==True:
    bn = np.apply_along_axis(BrigthnessNormalization, 0, r2)
    r2 = reshape_as_image(bn) 
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


####
pca.explained_variance_ratio_

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
=======
>>>>>>> ba9130e866e4585d8ff0c942a4ea9959de77704a
