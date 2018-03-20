# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:25:28 2018

@author: Lopatin
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

import rasterio
from sklearn.cross_decomposition import PLSRegression
from sklearn import model_selection 
from sklearn.metrics import mean_squared_error, make_scorer

### Input variaböes
#inData = "test_data.csv"
#yVariable = "BM"
#raster = "hyper_P1_5m.tif"
Boots = 500
    
def tune_PLSR(x, y):
    """ Parameter tuning of PLS regression """
    n_comp_range = range(1, int(maxComp))   
    param_grid = dict(n_components=n_comp_range)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    # Leave-one-out cross validation    
    cv = model_selection.LeaveOneOut()
    cv.get_n_splits(x)
    # grid search
    grid = model_selection.GridSearchCV(PLSRegression(), param_grid=param_grid, scoring=scorer, cv=cv)
    grid.fit(x, y)    
    scores = grid.grid_scores_ 
    return grid, scores

def saveRaster(img, inputRaster):
    # Save TIF image to a nre directory of name MNF
    img2 = np.transpose(img, [2,0,1]) # get to (band, raw, column) shape 
    output = str(raster[:-4])+"_PLSR_Predict.tif"
    new_dataset = rasterio.open(output , 'w', driver='GTiff',
               height=inputRaster.shape[0], width=inputRaster.shape[1],
               count=img.shape[2], dtype=str(img.dtype),
               crs=inputRaster.crs, transform=inputRaster.transform)
    new_dataset.write(img2)
    new_dataset.close()

def unlist(x):
    """ unlist a nested list into a 1D array """
    y = np.array([item for sublist in x for item in sublist])
    return y

###################
if __name__ == "__main__":
    
   # create the arguments for the algorithm
    parser = argparse.ArgumentParser()

    parser.add_argument('-i','--inData', 
      help='Input raster', type=str, required=True)
    parser.add_argument('-y','--yVariable', 
      help='Name of response variable', type=str, default=True)
    parser.add_argument('-r','--raster', 
      help='Input raster stack of predictors', type=str, default=True)
              
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = vars(parser.parse_args())
     
    # data inputs
    inData = args["inData"]
    yVariable = args["yVariable"] 
    raster = args["raster"]
 
    # dataset for grid search
    data = pd.read_csv(inData)
    N = len(data)
    maxComp = N/2
    
    # model data
    x = data.drop([yVariable], axis=1).astype('float32')
    y = data[yVariable].astype('float32')
    
    # Load raster
    r = rasterio.open(raster)
    r2 = r.read() # transform to array
    img = np.transpose( r2, [1,2,0] ) # transpose to shape (nrow, ncol, nbands)
    numBands, nrow, ncol = r2.shape
    
    # transfor a 3D array into a 2D for to apply regressions
    r_data = np.reshape(img, (nrow*ncol, numBands))
    r_data = r_data.astype('float32') # 32 bits
    r_data = np.nan_to_num(r_data)
    
    # run model tuning
    PLS, scores = tune_PLSR(x, y)
    bestComp = PLS.best_index_ + 1
    print('Scores:')
    pprint(scores)
    print('The best N° of components is = ', bestComp)
    
    ### Iterative validation   
    obs = []
    pred = []
    r2 = []
    nRMSE = []
    maps = []
    
    for i in tqdm( range(Boots) ):
        # select random number    
        idx = np.random.choice(N, N, replace= True)
        idx2 = list(set(range(N)) - set(idx))
        # select samples using idx
        x_train = np.array(x.loc[idx, :]) 
        x_val   = np.array(x.loc[idx2, :])
        y_train = np.array(y[idx])
        y_val   = np.array(y[idx2])
        
        # PLSR model
        trainPLSR = PLSRegression(n_components = bestComp)
        trainPLSR.fit(x_train, y_train)
        
        # predict
        predictt = trainPLSR.predict(x_val)
        predictt = unlist( predictt )
        
        # predict to map
        mapp = trainPLSR.predict(r_data)
        mapp = unlist(mapp)
        
        # backtransform maps to 3D array
        mapp = mapp.reshape(img[:, :, 0].shape)        
       
        # get accuracies
        R2 = (np.corrcoef(predictt, y_val)[0,1])**2
        nrmse = (mean_squared_error(y_val, predictt)/(np.max(y_val)-np.min(y_val)))
    
        # Store results
        obs.append(y_val)
        pred.append(predictt)
        r2.append(R2)
        nRMSE.append(nrmse)
        maps.append(mapp)
    
    # Get model metrics
    median_r2 = np.median(r2)
    median_nRMSE = np.median(nRMSE)
    print("Median r2 and nRMSE values are ", median_r2, median_nRMSE)
    metrics = pd.DataFrame({ 'r2' : r2, 
                             'nRMSE' : nRMSE})
    metrics.to_csv('metrics'+str(inData[:-4])+'.csv') # save
    
    # plot metrics distribution
    fig = plt.figure()
    plt.boxplot( [r2, nRMSE])
    plt.xticks([1, 2], ['r2', 'nRMSE'])
    plt.ylim((0,1))
    plt.title("Distribution of accuracies")
    fig.savefig("dist_accuracies_"+str(inData[:-4])+".pdf")
    
    # predicted maps stack
    stack_maps = np.dstack(maps)
    # save raster stack
    saveRaster(stack_maps, r)
    
    # median map
    median_map = np.apply_along_axis(np.median, 2, stack_maps)
    CV_map = np.apply_along_axis(np.std, 2, stack_maps)/np.mean(y)
    
    # maps values in 1D
    flat1 = unlist(np.reshape(median_map, (nrow*ncol, 1)))
    flat2 = unlist(np.reshape(CV_map, (nrow*ncol, 1)))
    
    # plot rasters
    fig = plt.figure(figsize=(12, 5))
    a = fig.add_subplot(1,2,1)
    plt.imshow(median_map, clim=(np.percentile(flat1, 5), np.percentile(flat1, 95)), 
               cmap = 'nipy_spectral')
    plt.colorbar()
    plt.title("Median pixel values of " + yVariable)
    a = fig.add_subplot(1,2,2)
    plt.imshow(median_map, clim=(np.percentile(flat2, 5), np.percentile(flat2, 95)),
               cmap="hot")
    plt.colorbar()
    plt.title("Coeff. variation pixel values of " + yVariable)
    
    fig.savefig( "predictedMaps.pdf" )




