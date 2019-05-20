#! /usr/bin/env python

################################################################################
'''
 TSA.py

 Program to perfor Time Series Analysis over a stack of yearly data (e.g. NDVI).
 Available estimates include linear tendency and Mann-Kandel Test

 @Author: Javier Lopatin
 @ Email: javierlopatin@gmail.com
 @ Last revision: 18/05/2019
 @ Version: 1.0

 Usage:

 python TSA.py -i <Input raster with yearly data>
               -o <Output raster with trends>
               -j <Number of parallel jobs to use during process [default = 4]>

 Examples:

 python TSA.py -i NDVI_timeSeries.tif -o Trends.tif -j 6

 The program is based in a scripts obtained at: https://www.uni-goettingen.de/en/524376.html
 I addapted the program to read big raster images in chuncks (blocks) of small
 size to keep the CPU mamory low. Plus, parallel processing is implemented.
 Speed if up to X20 tomes faster than the original script when using 4 cores.

'''
################################################################################

import rasterio
import numpy as np
from scipy import stats
import concurrent.futures

#####################################################
################## FUNCTIONS ########################
#####################################################

def mk_test(x, alpha=0.05):
    '''
    Mann-Kendall-Test
    Originally from: http://www.ambhas.com/codes/statlib.py
    I've changed the script though, now its about 35x faster than the original (depending on time series length)
    Input x must be a 1D list/array of numbers
    '''
    n = len(x)
    # calculate S
    listMa = np.matrix(x)  # convert input List to 1D matrix

    # calculate all possible differences in matrix
    subMa = np.sign(listMa.T - listMa)

    # with itself and save only sign of difference (-1,0,1)
    # sum lower left triangle of matrix
    s = np.sum(subMa[np.tril_indices(n, -1)])

    # calculate the unique data
    # return_counts=True returns a second array that is equivalent to tp in old version
    unique_x = np.unique(x, return_counts=True)
    g = len(unique_x[0])

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:  # there are some ties in data
        tp = unique_x[1]
        var_s = (n * (n - 1) * (2 * n + 5) +
                 np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    # calculate the p_value
    p = 2 * (1 - stats.norm.cdf(abs(z)))  # two tail test
    h = abs(z) > stats.norm.ppf(1 - alpha / 2)

    return h, p


def TSA(dstack):
    '''Function that calculates linear regression and Mann-Kendall-pValue coefficients
    for each raster pixel against continous time steps.
    Input must be an array with shape [rows, columns, bands] or a string with the
    full path to the file.
    '''
    # if funciton called from terminal:
    if __name__ == "__main__":
        # get to (raw, column, band) shape
        dstack = np.transpose(dstack, [1, 2, 0])

    # equally spaced time steps by length of inList
    timeList = np.asarray(list(range(len(dstack))))
    stepLen = len(dstack)

    # stack to 1D array
    dstack1D = dstack.reshape(-1)

    # Break down dstack1D into a list, each element in list contains the single steps
    # of one pixel -> List length is equal to number of pixels
    # List can be used to use Pythons map() function
    dstackList = [dstack1D[i:i + stepLen]
                  for i in range(0, len(dstack1D), stepLen)]

    # initialise empty arrays to be filled by output values, arrays are 1D
    slopeAr, intcptAr, rvalAr, pvalAr, stderrAr, mkPAr = [
        np.zeros(dstack[0].reshape(-1).shape) for _ in range(6)]

    # Use map() to iterate over each pixels timestep values for linear reg and Mann.Kendall
    # map(function_to_apply, list_of_inputs)
    # lambda function is a small anonymous function. Can have many arguments, but one expression
    # lambda arguments : expression
    # Method is about 10% faster than using 2 for-loops (one for x- and y-axis)
    # lineral tendency
    outListReg = list(
        map((lambda x: stats.linregress(timeList, x)), dstackList))
    # Mann-Kandel Test
    outListMK = list(map((lambda x: mk_test(x)), dstackList))

    for k in range(len(outListReg)):
        slopeAr[k] = outListReg[k][0]
        intcptAr[k] = outListReg[k][1]
        rvalAr[k] = outListReg[k][2]
        pvalAr[k] = outListReg[k][3]
        stderrAr[k] = outListReg[k][4]

        mkPAr[k] = outListMK[k][1]

    outShape = dstack[0].shape
    outTuple = (slopeAr.reshape(outShape),
                intcptAr.reshape(outShape),
                rvalAr.reshape(outShape),
                pvalAr.reshape(outShape),
                stderrAr.reshape(outShape),
                mkPAr.reshape(outShape))

    outStack = np.dstack(outTuple)  # stack results
    # get to (band, raw, column) shape
    outStack = np.transpose(outStack, [2, 0, 1])

    return outStack


def single_process(infile, outfile):
    '''
    Process infile in one-step. Use this with small
    raster images (low memory use).
    '''

    # open infile and change metadata
    with rasterio.open(infile) as src:
        profile = src.profile
        profile.update(count=6, dtype='float64')
        dstack = src.read()

    # fun TSA
    tsa = TSA(dstack)

    # save results
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(tsa)



def parallel_process(infile, outfile, n_jobs):
    """
    Process infile block-by-block with parallel processing
    and write to a new file.
    """
    from tqdm import tqdm # progress bar

    with rasterio.Env():

        with rasterio.open(infile) as src:

            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            profile = src.profile
            profile.update(blockxsize=128, blockysize=128,
                           count=6, dtype='float64', tiled=True)

            with rasterio.open(outfile, "w", **profile) as dst:

                # Materialize a list of destination block windows
                # that we will use in several statements below.
                windows = [window for ij, window in dst.block_windows()]

                # This generator comprehension gives us raster data
                # arrays for each window. Later we will zip a mapping
                # of it with the windows list to get (window, result)
                # pairs.
                data_gen = (src.read(window=window) for window in windows)

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_jobs
                ) as executor:

                    # We map the TSA() function over the raster
                    # data generator, zip the resulting iterator with
                    # the windows list, and as pairs come back we
                    # write data to the destination dataset.
                    for window, result in zip(
                        tqdm(windows), executor.map(TSA, data_gen)
                    ):
                        dst.write(result, window=window)
def main(infile, outfile, n_jobs):
    '''
    Check for the size of infile. if file is below 16384 observation [128 X 128].
    If below, use single_process; if above use parallel_process.
    '''
    with rasterio.open(infile) as src:
        width = src.width
        height = src.height

        if width*height <= 250000:
            single_process(infile, outfile)
        else:
            parallel_process(infile, outfile, n_jobs)



    #infile='/home/javier/Documents/SF_delta/Sentinel/TSA/test_year.tif'

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Time Series analysis with parallel raster processing")
    parser.add_argument('-i', '--inputImage',
                        help='Input raster with yearly time series', type=str)
    parser.add_argument('-o', '--outputImage',
                        help='Output raster with trend analysis', type=str)
    parser.add_argument('-a', '--alpha',
                        help='Alpha level of significance for Mann-Kandel [default = 0.05]', type=float, default=0.05)
    parser.add_argument(
        "-j",
        metavar="NUM_JOBS",
        type=int,
        default=4,
        help="Number of concurrent jobs [default = all available]",
    )
    args = parser.parse_args()

    main(args.inputImage, args.outputImage, args.j)
