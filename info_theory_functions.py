# information-theory-toolbox functions

import numpy as np
from scipy import *
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma

        
# in future want to incorporate bias estimators, and add a different way of calculating mutual information without adding and subtracting entropies.

# so in general, the histogram approach has more bias but less variance than the nearest-neighbor approach.  Memory also severely limits the applicability of the histogram approach - with just 10e3 samples, calculating the entropy of more than 8 dimensions becomes infeasible for modern consumer computers.

def entropy(x, Bins=10):
    '''Function to compute the mutual information between random variables x and y.  
    You need to specify the number of bins as well as the minimum and maximum of x and y (by default these
    are just the minimums and maximums of x and y).
    
    x should be samples by dimensions.  x also needs to be an array, so you could use asarray(x) 
    if you encounter errors.

    Bins can either be the number of bins (either one number to apply to all dimensions, or a sequence
    of nBins for each dimension), or can be a sequence of arrays of the bin edges along each dimension.
    
    I have also found that the smoothness of the estimate becomes better with more bins, since
    you get artifacts when a random variable moves into another bin whenever you sample.'''
    
    
    Counts, edges = np.histogramdd(x, bins=Bins)
    Probs         = Counts.astype(float)/float(sum(Counts))
    
    if abs(1 - sum(Probs)) > 0.01:
        print 'Probabilities do not sum to one ' + str(sum(Probs))
    
    H = 0.0
    for p in Probs.flat:
        if p != 0:
            H = H - p * log2(p)
            
    return H


def mutualinfo(x, y, Bins=10):
    '''Function to compute the mutual information between random variables x and y.  
    You need to specify the number of bins as well as the minimum and maximum of x and y (by default these
    are just the minimums and maximums of x and y).
    
    x and y should be samples by dimensions.  They also need to be an array, so you could use asarray(x) 
    if you encounter errors.

    Bins can either be the number of bins (either one number to apply to all dimensions, or a sequence
    of nBins for each dimension), or can be a sequence of arrays of the bin edges along each dimension.

    I have also found that the smoothness of the estimate becomes better with more bins, since
    you get artifacts when a random variable moves into another bin whenever you sample.'''
    
    if len(x.shape) < 2:
        x = np.reshape(x, (x.shape[0],1))
    if len(y.shape) < 2:
        y = np.reshape(y, (y.shape[0],1))


    return entropy(x, nBins) + entropy(y, nBins) - entropy(np.concatenate([x,y],axis=1), nBins)



def binaryWordsInformation(spikes,stimulus):
    '''Compute entropy of spike trains with binary words approach.
    
    Spikes and stimulus are both 1-d vertical numpy arrays
    with as many elements as neurons.
    '''
    nBins     = 2
    spikeBins = np.linspace(0,1,nBins+1)
    stimBins  = np.linspace(min(stimulus),max(stimulus),nBins+1)


    return mutualinfo(spikes, stimulus, Bins=[spikeBins, stimBins])


def getrho(x):
    if len(x.shape) < 2:
        x = np.reshape(x, (x.shape[0],1))
    D = squareform(pdist(x, 'euclidean'))
    D = D + np.max(D)*eye(D.shape[0])
    return np.min(D, axis=0)


def nnEntropy(x):
    '''Compute the binless entropy (bits) of a random vector using average nearest 
    neighbors distance (Kozachenko and Leonenko, 1987).
    
    For a review see Beirlant et al., 2001 or Chandler & Field, 2007.
    
    x is samples by dimensions.
    '''

    if len(x.shape) > 1:
        k = x.shape[1]
    else:
        k = 1
    
    Ak  = (k*pi**(float(k)/float(2)))/gamma(float(k)/float(2)+1)
    rho = getrho(x)
    
    return k*mean(log2(rho)) + log2(x.shape[0]*Ak/k) + log2(e)*0.5772156649


def nnInfo(x,y):
    '''nnInfo calculates mutual information between x and y 
    by using nearest neighbors entropy and I = HX + HY - HXY.
    '''
    
    if len(x.shape) < 2:
        x = np.reshape(x, (x.shape[0],1))
    if len(y.shape) < 2:
        y = np.reshape(y, (y.shape[0],1))
    
    return nnEntropy(x) + nnEntropy(y) - nnEntropy(np.concatenate([x,y],axis=1))
