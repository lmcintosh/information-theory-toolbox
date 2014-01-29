# information-theory-toolbox functions

import numpy as np
from scipy import *

# in future want to incorporate bias estimators, and add a different way of calculating mutual information without adding and subtracting entropies.

def mutualinfo(x, y, nBins, minX=0, maxX=0, minY=0, maxY=0):
    '''Function to compute the mutual information between random variables x and y.  
    You need to specify the number of bins as well as the minimum and maximum of x and y (by default these
    are just the minimums and maximums of x and y).
    
    x and y also need to be an array, so you could use asarray(x) if you encounter errors.
    
    I have also found that the smoothness of the estimate becomes better with more bins, since
    you get artifacts when a random variable moves into another bin whenever you sample.'''
    
    # in future want to make a default for min and max
    if minX==maxX:
        binsX = np.linspace(min(x), max(y), nBins+1)
    else:
        binsX = np.linspace(minX, maxX, nBins+1)
    if minY==maxY:
        binsY = np.linspace(min(y), max(y), nBins+1)
    else:
        binsY = np.linspace(minY, maxY, nBins+1)
    
    # Counting
    CountsXY, xedges, yedges = np.histogram2d(np.squeeze(x),np.squeeze(y), bins=[binsX, binsY])
    CountsX = sum(CountsXY,0) # this sums all the rows, leaving the marginal distribution of x
    CountsY = sum(CountsXY,1) # this sums all the cols, leaving the marginal distribution of y
    
    pX  = CountsX.astype(float)/float(sum(CountsX))
    pY  = CountsY.astype(float)/float(sum(CountsY))
    pXY = CountsXY.astype(float)/float(sum(CountsXY))
    
    
    if abs(1 - sum(pX)) > 0.01:
        print 'Probabilities pX do not sum to one ' + str(sum(pX))
    if abs(1 - sum(pY)) > 0.01:
        print 'Probabilities pY do not sum to one ' + str(sum(pY))
    if abs(1 - sum(pXY)) > 0.01:
        print 'Probabilities pXY do not sum to one ' + str(sum(pXY))
    
    
    # Entropies
    HX  = 0.0
    HY  = 0.0
    HXY = 0.0
    
    for prob in pY:
        if prob != 0:
            HY = HY - prob*log2(prob)
    
    for j in xrange(len(pX)):
        if pX[j] != 0:
            HX = HX - pX[j] * log2(pX[j])
            for i in xrange(len(pY)):
                if pXY[i,j] != 0:
                    HXY = HXY - pXY[i,j] * log2(pXY[i,j])
                    
    H = [HX, HY, HXY]
    I = HX + HY - HXY
    
    return H, I



def binaryWordsInformation(spikes,stimulus):
    '''Compute entropy of spike trains with binary words approach.
    
    Spikes and stimulus are both 1-d vertical numpy arrays
    with as many elements as neurons.
    '''
    nBins = 2
    H, I  = mutualinfo(spikes,stimulus,nBins,0,1,min(stimulus),max(stimulus))
    
    return I


def nearestNeighborsEntropy(x):
    '''Compute the binless entropy of a random vector using average nearest 
    neighbors distance (Kozachenko and Leonenko, 1987).

    '''

    rho = np.zeros((len(x),))
    H   = 0
    for idx,pt in enumerate(x):
        # get second smallest minimum distance (first will always be i=j)
        rho[idx] = sort(abs(pt - x[:idx]))[1]
        
        # H = 1/n sum_i=1^n ln(n*rho_n,i) + ln(2) + Euler's constant
        H += log((idx+1)*rho[idx])
    

    return (1.0/float(len(x)))*H + log(2) + 0.5772156649
