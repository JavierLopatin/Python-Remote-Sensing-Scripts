#! /usr/bin/env python

########################################################################################################
#
# apply_cluster.py
#
# A python script to perform apply KMeans clustering on Big Data Remote Sensing Data
#
#
# Author: Javier Lopatin
# Email: javier.lopatin@uai.cl
# Last changes: 13/07/2022
# Version: 1-0
#
# Parameters:
#            -i: input raster data [str]
#            -m: pre-trained KMeans model (see train_cluster.py) [str]
#            -j: n_jobs, number of parallel jobs to run [int] [default=4]
#            -c: chunckSize, size of the chuncks to split the original Big Data raster [int] [default=150; i.e., 150X150 pixeles]
#
# example:
#           python apply_cluster.py -i raster.tif -m model.sav [with default -j and -c]
#           python apply_cluster.py -i raster.tif -m model.sav -j 2 -c 100
#
########################################################################################################


import os
import argparse
import glob
import multiprocessing
import rasterio
import pickle
import numpy as np
import xarray as xr
import rioxarray as rio
from rioxarray.merge import merge_arrays
from tqdm import tqdm
from shutil import rmtree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def process(X, model, save=True):
    '''
    Function to apply the pre-trained KMeans algorithm

    '''
    if isinstance(X, str):
        img = rio.open_rasterio('temp/' + X)  # , chunks={'y':126, 'x':126})
    else:
        img=X
    img.values = img.values.astype('float64')
    img_flat = img.stack(z=('y', 'x'))
    img_flat = img_flat.transpose('z', 'band')
    if np.isnan(img_flat.values).any():  # if data contain NaN
        try:
            id = np.isnan(img_flat).any(axis=1)
            # predict only in clean data
            predict = model.predict(img_flat[~id, :])
            out = np.empty((img_flat.shape[0],))
            out[:] = np.nan
            out[~id] = predict  # fill emtpy array with predicted data
            output_array = img_flat[:, 0]  # copy xarray shape and metadata
            output_array = output_array.copy(data=out)
            output_array = output_array.unstack()  # back to original XY shape
            output_array.attrs['long_name'] = 'prediction'
        except:  # if all data are NaN or if something goes wrong
            output_array = img[0, :, :]
            output_array.attrs['long_name'] = 'prediction'
    else:
        out = model.predict(img_flat)
        output_array = img_flat[:, 0]  # copy xarray shape and metadata
        output_array = output_array.copy(data=out)
        output_array = output_array.unstack()  # back to original XY shape
        output_array.attrs['long_name'] = 'prediction'
    if save == True:
        output_array.rio.to_raster('out/' + X, dtype=np.float64)
    else:
        return output_array


if __name__ == "__main__":

    # create the arguments for the algorithm
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Imput raster',
                        type=str, required=True)
    parser.add_argument('-m', '--model', help='Imput model',
                        type=str, required=True)
    parser.add_argument(
        '-j', '--n_jobs', help='Number of parallel jobs', type=int, default=4)
    parser.add_argument(
        '-c', '--chunckSize', help='Size of chunks to work with. Must be multiple of 16', type=int, default=150)
    parser.add_argument('--version', action='version', version='%(prog)s 2.0')
    args = vars(parser.parse_args())

    # input raster
    # os.chdir('/home/javierlopatin/Documentos/temp/Siusun')
    inData = args["input"]
    inModel = args["model"]
    n_jobs = args["n_jobs"]
    chunckSize = args["chunckSize"]

    print('')
    print('#----------------------------------------------#')
    print('Chosen setup')
    print('#----------------------------------------------#')
    print('')
    print('n_jobs = ' + str(n_jobs))
    print('chunckSize = '+str(chunckSize)+'X'+str(chunckSize)+' pixeles')
    print('')

    # load pre-trained model
    model=pickle.load(open(inModel, 'rb'))

    # create temp folder
    if not os.path.exists("temp"):
        os.makedirs("temp")
    if not os.path.exists("out"):
        os.makedirs("out")
    # split raster into chunks and save them in temp folder
    print('')
    print('#----------------------------------------------#')
    print('Splitting big raster into tiles (temp folder)')
    print('#----------------------------------------------#')
    print('')
    os.system('gdal_retile.py -ps ' + str(chunckSize) + ' ' + \
              str(chunckSize) + ' -targetDir temp/ ' + inData)

    # list of tif files
    imgs=os.listdir('temp/')

    # apply clustering with parallel processing
    print('')
    print('#----------------------------------------------#')
    print('Apply KMeans in parallel')
    print('#----------------------------------------------#')
    print('')
    pool=multiprocessing.Pool(n_jobs)
    for _ in tqdm(pool.imap_unordered(process, imgs), total=len(imgs)):
        pass

    # list outputs
    imgs=os.listdir(path='out/')

    # merge data using rioxarray
    print('Merging all tiles...')
    all=glob.glob('out/*.tif')
    out=[]
    for r in all:
        out.append(rio.open_rasterio(r))
    output=merge_arrays(out)
    output.rio.to_raster(inData[:-4] + '_cluster.tif')

    # delete temp and out folders
    print('Deleting temporal folders...')
    rmtree('temp')
    rmtree('out')
    print('Done!')
