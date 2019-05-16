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
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        return X/np.sqrt( np.sum((X**2), 0) ).astype('float32')

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

def RunCanupo(inData, scales, step):
    """
    Run Canupo function for four non-systematic scales.
    Then, transform the msc outputs to txt and rasterize it using LASTools

    - inData: text with the xyz LiDAR data
    - scales: enter the start, end, and step for the scales
    - resolution: resolution to export the rasters

    Brodu, N. and Lague, D. (2012). 3D terrestrial lidar data classification of
    complex natural scenes using a multi-scale dimensionality criterion:
    Applications in geomorphology. ISPRS Journal of Photogrammetry and Remote
    Sensing, vol. 68, p.121-134.
    """
    import os, glob, shutil
    import numpy as np
    from subprocess import call

    # check for the direction of LASTools and CANUPO in your PC
    gdalDir = "C:/OSGeo4W64/bin/"
    lastoolsDir = "C:/lastools/bin/"

    # create temporal folder
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    # scales
    i0 = float(scales[0])
    i2 = float(scales[1])
    dif = float(scales[2])

    # run canupo
    outName = "tmp/out.msc"
    process = "canupo "+str(i0)+":"+str(dif)+":"+str(i2)+" : "+inData+" "+inData+" "+outName
    call(process)

    # msc2txt
    process = "msc_tool xyz "+outName+" : "+outName[:-4]+".txt"
    call(process)

    """
    Reorder the outputs, separate the components, and rasterize them
    Then, make a raster stack with the outputs
    """
    # variables to use
    components = np.arange(i0, i2+(dif), dif)
    N = len(components)
    nonUsed = 3 + N*3
    colList = range(nonUsed, nonUsed+N,1)

    # load original coordinated
    df = np.loadtxt("tmp/out.txt", usecols=[0,1])
    # load components
    df2 = np.loadtxt("tmp/out.txt", usecols=colList)
    # merge
    df3 = np.append(df, df2, axis=1)

    # loop through components
    for i in range(N):
        # export results
        out = df3[:, (0,1,i+2)]
        outName = "tmp/"+inData[:-4]+"_comp_"+str(i+1)+".txt"
        np.savetxt(outName, out)
        # rasterize with lasgrid
        process = lastoolsDir+"lasgrid.exe -i "+outName+" -o "+outName[:-4]+".tif -step "+str(step)+" -elevation -average"
        call(process)

    # stack bands
    outName = inData[:-4]+"_"+str(i0)+"_"+str(i2)+".tif"
    tif_list = glob.glob("tmp/*.tif")
    tif_list = " ".join(tif_list)
    process = "python "+gdalDir+"gdal_merge.py -o "+outName+" "+tif_list+" -separate"
    call(process)

    # return information of the created raster
    call("gdalinfo " + outName)

    # delate tables from memory
    del df, df2, df3

    # erase temporal folder
    shutil.rmtree("tmp")

###############

def Cloudmetrics2Raster(lidar, input_shp, ID):
    """
    Rasterize a LiDAR metrics estimated by CloudMetrics-FUSION
    It needs: the shapefile is a grid of polygons of the size of the raster pixel

    - lidar: point cloud information
    - input_shp: shapefile with continuos grid of polygons with the output pixels
    - ID: shapefile id name to use

    """
    import os, glob, math, rasterio
    from subprocess import call
    import pandas as pd
    import geopandas as gpd
    from tqdm import tqdm

    ### path and name of the input shapefile
    FusionDir = "C:/FUSION/"
    lastoolsDir = "C:/lastools/bin/"

    ### load shapefile
    r = gpd.read_file(input_shp)
    crs = r.crs # get CRS

    # get the position of the ID column in the attribute table
    idName = str
    for i in range(len(r.columns)):
        if r.columns[i] == ID:
            idName = i
            break
        else:
            continue

    # Get the ID values/names
    GetIdNames = r[idName]

    # create a tamporal folder to store intermediate files
    if not os.path.exists("FUSION_tmp"):
        os.makedirs("FUSION_tmp")

    ### Create cloudmetric file
    print("Creating Cloumetric file...")
    for i in tqdm( range(len(r)) ):
        outname = "FUSION_tmp/"+str(GetIdNames[i])
        # Save each shapefile
        shape = r[i:(i+1)]
        shape.to_file(outname + ".shp", driver='ESRI Shapefile')
        # Cut the point cloud using lastools
        process = lastoolsDir+"lasclip -i "+lidar+" -poly "+outname+".shp -o "+outname+".las"
        call(process)
        # Create cloudmetrics for each .las file
        process = FusionDir+"cloudmetrics "+outname+".las"+" Cloudmetrics.csv"
        call(process)
        # delete files from memory
        files = glob.glob('FUSION_tmp/*')
        for f in files:
            os.remove(f)
    print("Done!")

    ### Load the .csv file
    columns = [1] + list(range(13, 51))
    metrics = pd.read_csv("CloudMetrics.csv", usecols=columns)
    # replace spaces by "_" in the columnames
    metrics.columns = metrics.columns.str.replace(" ", "_")
    # replece the column mane 'FileTitle' with the shapefile ID
    metrics = metrics.rename(columns = {'FileTitle' : ID})

    ### merge the metrics with the shapefile
    r = r.merge(metrics, on=ID)
    out_shp = input_shp[:-4]+"_metrics.shp"
    # save shapefile
    r.to_file(out_shp, driver='ESRI Shapefile')

    ### creating metric raster
    # load the shapefile (important as the column names may have been shortened)
    r = gpd.read_file(out_shp)
    names = r.columns[-39:] # get column names
    if 'geometry' in names:
        names = names[:-1]
    pixel_size = pixel_size = math.sqrt(r.area[0]) # get pixel size
    r.crs = crs # set CRS

    ### rasterize all metrics
    print("Rasterizing the metrics...")
    for i in tqdm( range(len(names)) ):
        process = "gdal_rasterize -a "+names[i]+" -tr "+str(pixel_size)+" "+str(pixel_size)+" -l "+out_shp[:-4]+" "+out_shp+" "+"FUSION_tmp/"+names[i]+".tif"
        call(process)
    print("Done!")

    ### Stack rasters
    # Read metadata of first file
    with rasterio.open("FUSION_tmp/"+names[0]+".tif") as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(names))

    # Read each layer and write it to stack
    outName = input_shp[:-4]+"_metrics.tif"
    with rasterio.open(outName, 'w', **meta) as dst:
        for id, layer in enumerate(names):
            with rasterio.open("FUSION_tmp/"+layer+".tif") as src1:
                dst.write_band(id + 1, src1.read(1))

##################



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
    import shapefile, rasterio
    from rasterstats import zonal_stats
    import numpy as np
    # Raster management
    with rasterio.open(raster) as r:
        affine = r.affine
        bands = r.count
        bandNames = []
        for i in range(bands):
            a = "B" + str(i+1)
            bandNames.append(a)

    # empty matrix to store results
    matrix = np.empty((len(shapefile.Reader(shp).records()), bands))

    # Extract values
    for i in range(1, bands):
        # stats
        array = r.read(i+1) # open one band at a time
        stats = zonal_stats(shp, array, affine=affine, stats=func)
        matrix[:,i] = stats

    return matrix

##########

def ExtractPointValues(raster, shp):
    """ Extract raster values by a shapefile point mask.
    """
    import shapefile, rasterio
    from rasterstats import point_query
    import numpy as np
    # Raster management
    with rasterio.open(raster) as r:
        affine = r.affine
        bands = r.count
        bandNames = []
        for i in range(bands):
            a = "B" + str(i+1)
            bandNames.append(a)

    # empty matrix to store results
    matrix = np.empty((len(shapefile.Reader(shp).records()), bands))

    # Extract values
    for i in range(bands):
        # stats
        array = r.read(i+1) # open one band at a time
        stats = point_query(shp, array, affine=affine)
        matrix[:,i] = stats

    return matrix

##########

def setBandName(inputFile, band, name):
    """
    A function to set the no data value
    for each image band.
    """
    import osgeo.gdal as gdal
    # Open the image file, in update mode
    # so that the image can be edited.
    dataset = gdal.Open(inputFile, gdal.GA_Update)
    # Check that the image  has been opened.
    if not dataset is None:
        # Get the image band
        imgBand = dataset.GetRasterBand(band)
        # Check the image band was available.
        if not imgBand is None:
            # Set the image band name.
            imgBand.SetDescription(name)
        else:
            # Print out an error message.
            print("Could not open the image band: ", band)
    else:
        # Print an error message if the file
        # could not be opened.
        print("Could not open the input image file: ", inputFile)

###########

def reproject_image_to_master ( master, slave, res=None ):
    """This function reprojects an image (``slave``) to
    match the extent, resolution and projection of another
    (``master``) using GDAL. The newly reprojected image
    is a GDAL VRT file for efficiency. A different spatial
    resolution can be chosen by specifyign the optional
    ``res`` parameter. The function returns the new file's
    name.
    Parameters
    -------------
    master: str
        A filename (with full path if required) with the
        master image (that that will be taken as a reference)
    slave: str
        A filename (with path if needed) with the image
        that will be reprojected
    res: float, optional
        The desired output spatial resolution, if different
        to the one in ``master``.
    Returns
    ----------
    The reprojected filename
    TODO Have a way of controlling output filename
    """
    import gdal
    slave_ds = gdal.Open( slave )
    if slave_ds is None:
        raise IOError
    slave_proj = slave_ds.GetProjection()
    slave_geotrans = slave_ds.GetGeoTransform()
    data_type = slave_ds.GetRasterBand(1).DataType
    n_bands = slave_ds.RasterCount

    master_ds = gdal.Open( master )
    if master_ds is None:
        raise IOError
    master_proj = master_ds.GetProjection()
    master_geotrans = master_ds.GetGeoTransform()
    w = master_ds.RasterXSize
    h = master_ds.RasterYSize
    if res is not None:
        master_geotrans[1] = float( res )
        master_geotrans[-1] = - float ( res )

    dst_filename = slave.replace( ".tif", "_crop.tif" )
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( master_geotrans )
    dst_ds.SetProjection( master_proj)

    gdal.ReprojectImage( slave_ds, dst_ds, slave_proj,
                         master_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None  # Flush to disk
    return dst_filename

############
