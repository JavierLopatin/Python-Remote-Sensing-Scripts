#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
En example of resample a raster using rioxarray

@author: javier
"""

import numpy as np
import rioxarray as xr
import matplotlib.pyplot as plt

r = 'Cau_19_02_05_MSR.tif'

# funciton to resample resolution using avarage


def resample(xarray, reduction):
    reduced = (
        xarray
        .groupby(((xarray.y//reduction) + 0.5) * reduction).mean(dim='y')
        .groupby(((xarray.x//reduction) + 0.5) * reduction).mean(dim='x'))
    reduced.rio.set_crs(xarray.rio.crs)
    return reduced


img = xr.open_rasterio(r)

img_5m = resample(img, 5)

img_5m = img_5m.rio.set_crs(img.rio.crs)
img_5m.rio.crs

NDVI = (img[4, :, :]-img[2, :, :])/(img[4, :, :]+img[2, :, :])
NDVI_5m = (img_5m[4, :, :]-img_5m[2, :, :])/(img_5m[4, :, :]+img_5m[2, :, :])

# cover
idx = NDVI > 0.2
binary = idx.astype('int16')

# downscaling to 5m pixels
cover_5m = resample(binary, 5)

# get nan mask
nan = np.isnan(NDVI_5m).values
# mask cover
cover_5m.values[nan] = np.nan

NDVI_5m.plot(robust=True)
cover_5m.plot(robust=True)

# plot comparison
ndvi = np.reshape(NDVI_5m.values, [NDVI_5m.values.shape[0]*NDVI_5m.values.shape[1]])
cover = np.reshape(cover_5m.values, [cover_5m.values.shape[0]*cover_5m.values.shape[1]])

ndvi = ndvi[~np.isnan(ndvi)]
cover = cover[~np.isnan(cover)]

corr = np.corrcoef(ndvi, cover)[1][0]

plt.plot(ndvi, cover, 'o', alpha=0.1)
plt.title('r = '+str(corr.round(2)), loc='left', size=15)
plt.ylabel('Fractional cover [%]')
plt.xlabel('NDVI')

# save resampled data
NDVI_5m.rio.to_raster('NDVI_5m.tif')
cover_5m.rio.to_raster('cover_5m.tif')

# other indices
WBI = img_5m[0, :, :]/img_5m[4, :, :]
NDWI = ((img_5m[1, :, :]-img_5m[4, :, :])/(img_5m[1, :, :]+img_5m[4, :, :]))

# save
WBI.rio.to_raster('WBI.tif')
NDWI.rio.to_raster('NDWI.tif')
