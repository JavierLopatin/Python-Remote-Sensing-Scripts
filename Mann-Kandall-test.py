
################################################################################
# Mann-Kendall-Test
#
# Originally from: http://www.ambhas.com/codes/statlib.py, and adapted at
# https://www.uni-goettingen.de/en/524376.html to improve speed
#
# Input x must be a 1D list/array of numbers
################################################################################

import numpy as np

def mk_test(x, alpha=0.05):

    n = len(x)

    # calculate S
    listMa = np.matrix(x)              # convert input List to 1D matrix
    subMa = np.sign(listMa.T - listMa) # calculate all possible differences in matrix
                                       # with itself and save only sign of difference (-1,0,1)
    s = np.sum(subMa[np.tril_indices(n, -1)]) # sum lower left triangle of matrix

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
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))  # two tail test
    h = abs(z) > scipy.stats.norm.ppf(1 - alpha / 2)

    return h, p
