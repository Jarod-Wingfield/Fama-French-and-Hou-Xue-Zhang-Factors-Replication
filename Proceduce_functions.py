##########################################
# Replication of factors                 #
# Wu Jiaying                             #
# Date: April 2025                       #
# Updated: April 8th 2025                #
##########################################
# Functions used in main procedure

import pandas as pd
import numpy as np
import scipy

def assign_interval(val, intervals):
    for iv in intervals:
        if val in iv:
            return iv
    return np.nan

# Function for sorting and assigning group labels.
def q_func(x ,sort_data, q, labels):
    # If all nan, no group
    if all(x.isna()):
        return x
    else:
        # Sorting data (pick date group)
        st=sort_data.get_group(x.name)

        # Quantiles point of sorting data
        try:
            qp=(pd.qcut(st,q).dtypes).categories.tolist()
        except:
            # no enough non-nan value to calculate quantiles
            x.iloc[:]=np.nan
            return x
        
        # change the interval min and max to fit the original data, and left closed
        
        for i in range(len(qp)):
            if i==0:
                qp[0]=pd.Interval(x.min(),qp[0].right,closed='left')
            elif i==len(qp)-1:
                qp[-1]=pd.Interval(qp[-1].left,x.max(),closed='both')
            else:
                qp[i]=pd.Interval(qp[i].left,qp[i].right,closed="left")


        # It is not allow to use both interval in the pd.Categorical step, which will make all value to np.nan.
        # Here is a problem of minimum obs.' group.
        # Assign intervals at first
        assigned_x = [assign_interval(val, qp) for val in x]

        # Assign group intervals
        f=pd.Categorical(assigned_x,qp)

        # Assign group labels
        f=pd.Series(f,index=x.index)
        f=f.cat.rename_categories(labels)

        return f
    

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan
    


# GRS test function
def GRS_test(factor, resid, alpha):
    '''
    factor: T*L
    resid: T*N
    alpha: N*1
    '''

    N = resid.shape[1]
    T = resid.shape[0]
    L = factor.shape[1]

    factor = np.asmatrix(factor)                  
    resid = np.asmatrix(resid)                     
    alpha = np.asmatrix(alpha).reshape(N, 1)       

    factor_return_mean = factor.mean(axis=0)

    # Residual covariance matrix
    cov_resid = (resid.T * resid) / (T-L-1)

    # Covariance matrix of factors
    cov_factor = ((factor - factor_return_mean).T *
                  (factor - factor_return_mean)) / (T-1)

    # Factors
    factor_return_mean = factor_return_mean.reshape(L, 1)

    # GRS statistic
    f_grs = float((T/N) * ((T-N-L)/(T-L-1)) * ((alpha.T * np.linalg.inv(cov_resid) * alpha) /
                  (1 + factor_return_mean.T * np.linalg.inv(cov_factor) * factor_return_mean)))

    # p-value
    p_grs = 1 - scipy.stats.f.cdf(f_grs, N, (T-N-L))

    return f_grs, p_grs