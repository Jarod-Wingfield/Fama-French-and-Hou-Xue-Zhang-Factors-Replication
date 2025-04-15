##########################################
# Replication of factors                 #
# Jiaying Wu                             #
# Date: April 2025                       #
# Updated: April 8th 2025                #
##########################################

# Import required libraries and modules
from Data_preprocess import Data_preprocess
from Proceduce_functions import assign_interval, q_func, wavg, GRS_test

# Standard library imports
from pandas.tseries.offsets import *
import pandas as pd
import os
import numpy as np
import datetime
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy

# Configure pandas to use Copy-on-Write mode
pd.options.mode.copy_on_write = True

##############################################
# Data Loading and Initial Processing        #
##############################################
# Initialize data loader
Data_loading=Data_preprocess(data_path='data', portfolio_month=6)

# Load Compustat (annual and quarterly)
comp, comp_quarterly=Data_loading.Compustat(file_annually='Compustat_CRSP_1960_2024/Compustat_Annually_1960_2024_part.csv', file_quarterly='Compustat_CRSP_1960_2024/Compustat_Quarterly_1960_2024.csv')

# Load CRSP data
crsp_monthly, crsp_month_win=Data_loading.CRSP(file_var='Compustat_CRSP_1960_2024/CRSP_Monthly_1960_2024.csv', file_name='Compustat_CRSP_1960_2024/NAME.csv', file_dl='Compustat_CRSP_1960_2024/Delisted Stocks 1960_2024.csv')

# Load CCM link table and link CRSP-Compustat
ccm_month_win, crsp_monthly=Data_loading.CCM(file='Compustat_CRSP_1960_2024/CCM_CRSP_Link_Table_CRSP.csv')

##############################################
# Sample Filtering and Sorting Preparation   #
##############################################

# Restrict to common stocks (share codes 10/11) on NYSE, AMEX, and NASDAQ
ccm_month_win = ccm_month_win[(ccm_month_win['exchange'].isin(['NYSE','AMEX','NASDAQ'])) & (ccm_month_win['SHRCD'].isin([10,11]))]
crsp_monthly = crsp_monthly[(crsp_monthly['exchange'].isin(['NYSE','AMEX','NASDAQ'])) & (crsp_monthly['SHRCD'].isin([10,11]))]

# -------------------------------------------
# Replication Targets:
# 1. Fama-French 6 Factors
# 2. q-Factors
# 3. Spanning Regression Table: Average Return, Alpha (6-factor and q-factor), GRS Test
# -------------------------------------------

# 1: Fama-French 6 factors

# In Fama-French factor construction:
# Stocks of firms with non-positive book equity are excluded 
# from BE/ME, OP, CP, and AC/BE sorts.

# ---------------------------------------------------------------
# 1. Fama-French 6 Factors
# ---------------------------------------------------------------

## ----------------------------
## HML (High minus Low)
## ----------------------------
# Value factor based on book-to-market equity (BE/ME)
# Constructed from independent 2x3 sorts:
# - Size groups: Small / Big (based on NYSE median market equity)
# - BE/ME groups: Low / Neutral / High (based on NYSE 30th and 70th percentiles)
# 
# Portfolio returns:
#   L_S, N_S, H_S (Small size)
#   L_B, N_B, H_B (Big size)
# 
# HML_S = H_S - L_S
# HML_B = H_B - L_B
# HML   = 0.5 * (HML_S + HML_B)

## ----------------------------
## CMA (Conservative minus Aggressive)
## ----------------------------
# Investment factor based on asset growth
# Same 2x3 sorting approach as HML
# Second sort based on the rate of growth in total assets (low to high)

## ----------------------------
## RMW_O (Robust minus Weak — Operating Profitability)
## ----------------------------
# Profitability factor based on operating profitability
# Operating profits are net of interest expense and scaled by book equity
# Constructed using the same independent 2x3 sorting logic as HML

## ----------------------------
## RMW_C (Cash Profitability)
## ----------------------------
# Cash profitability = operating profits minus accruals
# Scaled by book equity

## ----------------------------
## SMB (Small minus Big — Size Factor)
## ----------------------------
# Fama, French (2018) use SMB_C as SMB
#
# SMB_O and SMB_C are composite size factors based on multiple characteristics:
#   - SMB_BM (from HML sort)
#   - SMB_OP or SMB_CP (from RMW_O or RMW_C)
#   - SMB_Inv (from CMA sort)
#
# Example:
#   SMB_BM = (1/3) * (L_S + N_S + H_S) - (1/3) * (L_B + N_B + H_B)
#   SMB_O  = (1/3) * (SMB_BM + SMB_OP + SMB_Inv)
#   SMB_C  = (1/3) * (SMB_BM + SMB_CP + SMB_Inv)

# ---------------------------------------------------------------
# UMD (Up minus Down — Momentum Factor)
# ---------------------------------------------------------------
# Momentum portfolios updated monthly (not annually)
# Sorting is based on past returns:
#   - Mom = average return from month t−12 to t−2
#   - Portfolio formed at the end of month t−1


# Extract NYSE stocks grouped by portfolio month for ME and Investment factor breakpoints
NYSE_data=ccm_month_win[(ccm_month_win['exchange'].isin(['NYSE']))].groupby(['jdate'])
# Extract NYSE stocks with positive book equity, grouped by portfolio month, for BE/ME, OP, and CP breakpoints
NYSE_data_posi_be=ccm_month_win[(ccm_month_win['exchange'].isin(['NYSE']))&(ccm_month_win['be']>0)].groupby(['jdate'])

# Assign portfolio groups based on NYSE breakpoints:
# - Size (ME), split into Small (S) and Big (B) using NYSE median
# - Investment (Inv), B/M (beme), Operating Profitability (OP), and Cash Profitability (CP) sorted into Low (L), Neutral (N), and High (H)
ccm_month_win['ME_groups']=ccm_month_win.groupby(['jdate'], group_keys=False)['me'].apply(lambda x: q_func(x, sort_data=NYSE_data['me'], q=[0,0.5,1], labels=['S','B']))
ccm_month_win['Inv_groups']=ccm_month_win.groupby(['jdate'], group_keys=False)['Inv'].apply(lambda x: q_func(x, sort_data=NYSE_data['Inv'], q=[0,0.3,0.7,1], labels=['L','N','H']))
ccm_month_win['BM_groups']=ccm_month_win.groupby(['jdate'], group_keys=False)['beme'].apply(lambda x: q_func(x, sort_data=NYSE_data_posi_be['beme'], q=[0,0.3,0.7,1], labels=['L','N','H']))
ccm_month_win['OP_groups']=ccm_month_win.groupby(['jdate'], group_keys=False)['OP'].apply(lambda x: q_func(x, sort_data=NYSE_data_posi_be['OP'], q=[0,0.3,0.7,1], labels=['L','N','H']))
ccm_month_win['CP_groups']=ccm_month_win.groupby(['jdate'], group_keys=False)['CP'].apply(lambda x: q_func(x, sort_data=NYSE_data_posi_be['CP'], q=[0,0.3,0.7,1], labels=['L','N','H']))

# Define fiscal year used for July-to-June portfolio assignment
ccm_month_win['ffyear']=ccm_month_win['jdate'].dt.year

# Merge portfolio labels with monthly CRSP returns
df_monthly_fm=pd.merge(crsp_monthly, ccm_month_win[['PERMNO','ffyear']+list(set((ccm_month_win.columns).tolist()).difference(crsp_monthly.columns))],
                     how='left', on=['PERMNO','ffyear'])
df_monthly_fm=df_monthly_fm.sort_values(['PERMNO','date'])

# ----- Momentum (UMD) -----
# Sort for portfolio formed at the end of month t-1 is based on Mom, the average return from t-12 to t-2.
# To be included in a portfolio for month t (formed at the end of month t-1), a stock must have a price for the end of month t-13 and a good return for t-2. 
# In addition, any missing returns from t-12 to t-3 must be -99.0, CRSP's code for a missing price. 
# Each included stock also must have ME for the end of month t-1.

df_monthly_fm['retadj_for_mom']=df_monthly_fm['retadj'].fillna(-99.0)
df_monthly_fm['Momentum']=df_monthly_fm.groupby('PERMNO')['retadj_for_mom'].transform(lambda x: x.shift(2).rolling(window=11).sum()/11)

# Assign momentum groups based on NYSE breakpoints (30th and 70th percentiles)
NYSE_data=df_monthly_fm[(df_monthly_fm['exchange'].isin(['NYSE']))].groupby(['date'])
df_monthly_fm['Mom_groups']=df_monthly_fm.groupby(['date'], group_keys=False)['Momentum'].apply(lambda x: q_func(x, sort_data=NYSE_data['Momentum'], q=[0,0.3,0.7,1], labels=['L','N','H']))

# ----- Market Factor (MKT-RF) -----
# VW monthly return (all NYSE, Amex, and Nasdaq common stocks with a CRSP share code of 10 or 11)
# For the market factor, they do not exclude financial firms or firms with negative book equity.

# Load 1-month risk-free rate (RF) from Ken French library
RF=pd.read_excel('data/Fama French/F-F_RF_Monthly.xls')
# Adjust date to month-end
RF['date']=RF['date'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m'))+MonthEnd()

# Adjust date to month-end
df_monthly_fm['date']=df_monthly_fm['date']+MonthEnd(0)
# Compute value-weighted average return across all stocks (MKT)
FF_factors=df_monthly_fm.groupby(['date'],group_keys=True).apply(wavg, 'retadj','wt').reset_index()
FF_factors.columns=['date','MKT']

# Merge MKT with RF and compute excess market return
FF_factors=pd.merge(FF_factors,RF,on='date',how='left')
FF_factors['MKT-RF']=FF_factors['MKT']-FF_factors['RF']
# Compare with Fama-French 2*3 data, it is very similar but a litter difference.
# * This clarifies that the sample, weight are without systematic mistake.

# ----- Style Factors (HML, CMA, RMW, UMD) and Size-Factor Variants -----

# Define factor construction logic with corresponding grouping variable
factors_g=[['HML','BM_groups'],['CMA','Inv_groups'],['RMW_O','OP_groups'],['RMO_C','CP_groups'],['UMD','Mom_groups']]

# * Ken French library: 
# Stocks: Rm-Rf includes all NYSE, AMEX, and NASDAQ firms. 
# SMB, HML, RMW, and CMA for July of year t to June of t+1 include all NYSE, AMEX, and NASDAQ stocks for which we have market equity data for December of t-1 and June of t, (positive) book equity data for t-1 (for SMB, HML, and RMW), non-missing revenues and at least one of the following: cost of goods sold, selling, general and administrative expenses, or interest expense for t-1 (for SMB and RMW), and total assets data for t-2 and t-1 (for SMB and CMA).
# The six portfolios used to construct Mom each month include NYSE, AMEX, and NASDAQ stocks with prior return data. 
# To be included in a portfolio for month t (formed at the end of month t-1), 
# a stock must have a price for the end of month t-13 and a good return for t-2. 
# In addition, any missing returns from t-12 to t-3 must be -99.0, CRSP's code for a missing price. 
# Each included stock also must have ME for the end of month t-1.

# Create lagged columns for price (t-13) and return (t-2)
df_monthly_fm['price_t-13'] = df_monthly_fm.groupby('PERMNO')['PRC'].shift(13)
df_monthly_fm['retadj_t-2'] = df_monthly_fm.groupby('PERMNO')['retadj'].shift(2)

for i, g in enumerate(factors_g):
    # Apply inclusion criteria based on FF documentation
    if g[0] in ('HML'):
        stocks=df_monthly_fm[(df_monthly_fm['be']>0)&(df_monthly_fm['dec_me'].notna())&(df_monthly_fm['me'].notna())]
    elif g[0] in ('RMW_C','RMW_O'):
        stocks=df_monthly_fm[(df_monthly_fm['be']>0)&(df_monthly_fm['dec_me'].notna())&(df_monthly_fm['me'].notna())&(df_monthly_fm['revt'].notna())&(df_monthly_fm[['cogs', 'xsga', 'xint']].notna().any(axis=1))]
    elif g[0] in ('CMA'):
        stocks=df_monthly_fm[(df_monthly_fm['Inv'].notna())&(df_monthly_fm['dec_me'].notna())&(df_monthly_fm['me'].notna())]
    elif g[0] in ('UMD'):
        # stock must have a price for the end of month t-13 and a good return for t-2
        stocks=df_monthly_fm[(df_monthly_fm['me'].notna())&(df_monthly_fm['price_t-13'].notna())&(df_monthly_fm['retadj_t-2'].notna())&(df_monthly_fm['Momentum'].notna())]
    
    # Compute value-weighted return across 2x3 portfolios (size x characteristic)
    vw_g=stocks.groupby(['date','ME_groups',g[1]],group_keys=True).apply(wavg, 'retadj','wt').reset_index().rename(columns={0: 'vwret'})\
                    .pivot(index=['date'], columns=[g[1],'ME_groups'], values='vwret')

    # Rename MultiIndex
    vw_g.columns=[colname[0]+'_'+colname[1] for colname in vw_g.columns]
    
    # Compute HML, CMA, RMW, UMD based on 2x3 returns
    # e.g. Value-weighted monthly return, L_S, N_S, H_S, L_B, N_B, H_B. 
    # HML_S=H_S-L_S, HML_B=H_B-L_B, HML=1/2*(HML_S+HML_B)

    if g[0]=='CMA': # Reversal sign for investment factor
        # Low to High
        vw_g[g[0]+'_S']=vw_g['L_S']-vw_g['H_S']
        vw_g[g[0]+'_B']=vw_g['L_B']-vw_g['H_B']
    else:
        vw_g[g[0]+'_S']=vw_g['H_S']-vw_g['L_S']
        vw_g[g[0]+'_B']=vw_g['H_B']-vw_g['L_B']

    vw_g[g[0]]=1/2*(vw_g[g[0]+'_S']+vw_g[g[0]+'_B'])

    # Compute size premium for each factor
    # SMB_BM = 1/3(L_S + N_S + H_S) - 1/3 (L_B + N_B + H_B), SMB_OP, SMB_CP, SMB_Inv for SMB
    vw_g['SMB_'+g[1].split('_')[0]]=1/3*(vw_g['L_S']+vw_g['N_S']+vw_g['H_S']) - 1/3*(vw_g['L_B']+vw_g['N_B']+vw_g['H_B'])

    # Merge back to factor dataframe
    FF_factors=pd.merge(FF_factors,vw_g[[g[0],'SMB_'+g[1].split('_')[0]]],on='date',how='left')

    print(g)

# Aggregate SMB variants
# SMB_O = 1/3 (SMB_BM + SMB_OP + SMB_Inv)
# SMB_C = 1/3 (SMB_BM + SMB_CP + SMB_Inv)
FF_factors['SMB_O']=1/3*(FF_factors['SMB_BM']+FF_factors['SMB_OP']+FF_factors['SMB_Inv'])
FF_factors['SMB_C']=1/3*(FF_factors['SMB_BM']+FF_factors['SMB_CP']+FF_factors['SMB_Inv'])

# Restrict sample period
FF_factors=FF_factors[(pd.to_datetime(FF_factors.date)>=datetime.datetime.strptime("19670101",'%Y%m%d'))
        &(pd.to_datetime(FF_factors.date)<=datetime.datetime.strptime("20241231",'%Y%m%d'))]
FF_factors

# Load official FF factors for comparison
FF_factors_web=pd.read_excel('data/Fama French/F-F_Factors_Monthly.xls')
FF_factors_web['date']=FF_factors_web['date'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m'))+MonthEnd()
FF_factors_web.columns=['date','MKT-RF_FF','SMB_O_FF','HML_FF','RMW_O_FF','CMA_FF','RF_FF']

FF_factors_mom=pd.read_excel('data/Fama French/F-F_Mom_Monthly.xls')
FF_factors_mom['date']=FF_factors_mom['date'].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m'))+MonthEnd()
FF_factors_mom.columns=['date','UMD_FF']

FF_factors_web=pd.merge(FF_factors_web,FF_factors_mom,on='date',how='left')
FF_factors_all=pd.merge(FF_factors,FF_factors_web,on='date',how='left')

# Report correlation between constructed and official factors
for i in ['MKT-RF','SMB_O','HML','RMW_O','CMA','UMD']:
    print('Correlation between my construction factor '+i+' and Fama French factor:', FF_factors_all[i].corr(FF_factors_all[i+'_FF']))


# ---------------------------------------------------------------
# 2: q-factors
# ---------------------------------------------------------------

# Reminder on the q-factor methodology: The sample includes all common stocks from NYSE, AMEX, and NASDAQ
# with a CRSP share code of 10 or 11.
# Independent sorting:
# - A 2-by-3-by-3 sort is performed based on size, I/A (Investment-to-Assets ratio), and ROE (Return on Equity).
# Size: At the end of June each year (t), the median market equity of NYSE firms is used to split NYSE, AMEX, and NASDAQ stocks into two groups: small and large.
# I/A: At the end of June each year (t), the NYSE breakpoints are used to categorize stocks into three groups: the lowest 30%, middle 40%, and highest 30% based on I/A values.
# ROE: At the beginning of each month, the NYSE breakpoints categorize stocks into three groups based on ROE: the lowest 30%, middle 40%, and highest 30%.
# 
# Follow the methodology from Hou-Xue-Zhang: "Technical Document: Factors" (2025)

# Filter out financial firms and firms with negative book equity
ccm_month_win = ccm_month_win[((ccm_month_win['sic']<6000)|(ccm_month_win['sic']>6999)) & (ccm_month_win['be']>0)]

# Data for sorting based on NYSE stocks (used to calculate breakpoints)
NYSE_data_win=ccm_month_win[(ccm_month_win['exchange'].isin(['NYSE']))].groupby(['jdate'])

# Sorting by size, I/A, and ROE
ccm_month_win['Size_groups']=ccm_month_win.groupby(['jdate'], group_keys=False)['me'].apply(lambda x: q_func(x, sort_data=NYSE_data_win['me'], q=[0,0.5,1], labels=['S','B']))
ccm_month_win['IA_groups']=ccm_month_win.groupby(['jdate'], group_keys=False)['ItA'].apply(lambda x: q_func(x, sort_data=NYSE_data_win['ItA'], q=[0,0.3,0.7,1], labels=['L','N','H']))

# Assign portfolio group marks to the next year period (July to the following June)
ccm_month_win['ffyear']=ccm_month_win['jdate'].dt.year

# Merge portfolio groupings back with monthly stock return records
df_monthly_q=pd.merge(crsp_monthly, ccm_month_win[['PERMNO','ffyear']+list(set((ccm_month_win.columns).tolist()).difference(crsp_monthly.columns))],
                     how='left', on=['PERMNO','ffyear'])

# Exclude financial firms and firms with negative book equity
df_monthly_q=df_monthly_q[((df_monthly_q['sic']<6000)|(df_monthly_q['sic']>6999))]
df_monthly_q=df_monthly_q[df_monthly_q['be']>0]

df_monthly_q=df_monthly_q.sort_values(['PERMNO','date'])

# Monthly sorting based on ROE
NYSE_data_monthly=df_monthly_q[(df_monthly_q['exchange'].isin(['NYSE']))].groupby(['date'])
df_monthly_q['Roe_groups']=df_monthly_q.groupby(['date'], group_keys=False)['Roe'].apply(lambda x: q_func(x, sort_data=NYSE_data_monthly['Roe'], q=[0,0.3,0.7,1], labels=['L','N','H']))

# Construct 2-3-3 portfolios
vw_18g=df_monthly_q.groupby(['date','Size_groups','IA_groups','Roe_groups'],group_keys=True).apply(wavg, 'retadj','wt').reset_index().rename(columns={0: 'vwret'})

# Create pivot tables for size, investment, and ROE factors
vw_18g_size=vw_18g.pivot(index=['date'], columns=['Size_groups','IA_groups','Roe_groups'], values='vwret')
vw_18g_investment=vw_18g.pivot(index=['date'], columns=['IA_groups','Size_groups','Roe_groups'], values='vwret')
vw_18g_roe=vw_18g.pivot(index=['date'], columns=['Roe_groups','Size_groups','IA_groups'], values='vwret')

# Calculate the q-factors
ME_factors=pd.DataFrame(1/9*(vw_18g_size['S'].sum(axis=1)-vw_18g_size['B'].sum(axis=1))).reset_index().rename(columns={0: 'R_ME'})
IA_factors=pd.DataFrame(1/6*((vw_18g_investment['L'].sum(axis=1)-vw_18g_investment['H'].sum(axis=1)))).reset_index().rename(columns={0: 'R_IA'})
Roe_factors=pd.DataFrame(1/6*((vw_18g_roe['H'].sum(axis=1)-vw_18g_roe['L'].sum(axis=1)))).reset_index().rename(columns={0: 'R_ROE'})

# Combine the factors into a single DataFrame
q_factors=pd.merge(ME_factors,IA_factors,on='date',how='left')
q_factors=pd.merge(q_factors,Roe_factors,on='date',how='left')

# Load the q-factors from Hou-Xue-Zhang's website for comparison
q_factors_home=pd.read_csv('data/q factors/q5_factors_monthly_2024.csv')
q_factors_home['date']=pd.to_datetime(q_factors_home[['year', 'month']].assign(day=1)) + MonthEnd(0)
q_factors_home.drop(columns=['year','month'],inplace=True)
q_factors_home.columns=[i+'_q' for i in q_factors_home.columns[:-1]]+['date']

q_factors_all=pd.merge(q_factors,q_factors_home,on='date',how='left')

# Check the correlation between the constructed factors and those from Hou-Xue-Zhang
for i in ['R_ME','R_IA','R_ROE']:
    print('Correlation between my construction factor '+i+' and Hou-Xue-Zhang factor:', q_factors_all[i].corr(q_factors_all[i+'_q']))

# ---------------------------------------------------------------
# 3: Replicate the table for head-to-head factor spanning tests (Hou et al., 2019)
# ---------------------------------------------------------------

# Extract factors (Fama-French and q-factors)
all_factors=pd.merge(FF_factors,q_factors,on='date',how='inner')[['date','RF','MKT-RF','HML','SMB_O','RMW_O','CMA','UMD','R_ME','R_IA','R_ROE']]
# The market factor in the q-factor model is the same as in the Fama-French 6-factor model
all_factors['R_MKT']=all_factors['MKT-RF']
all_factors=all_factors.rename(columns={'MKT-RF':'MKT_RF'})
all_factors.to_csv('master/Factors_monthly.csv', index=False)

# Calculate Sharpe Ratios
Sharpe_ratio = []

for factor in ['MKT_RF','HML','SMB_O','RMW_O','CMA','UMD','R_ME','R_IA','R_ROE']:
    monthly_returns = all_factors[factor]
    
    # Annualize returns and volatility
    annualized_return = monthly_returns.mean() * 12
    annualized_vol = monthly_returns.std() * np.sqrt(12)
    
    # Calculate Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_vol
    
    # Append to results
    Sharpe_ratio.append({
        'Factor': factor,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio
    })

# Convert to DataFrame
Sharpe_ratio = pd.DataFrame(Sharpe_ratio)

# Save to CSV
Sharpe_ratio.to_csv('master/Factor_sharpe_ratios.csv', index=False)

print('Sharpe Ratio:\n',Sharpe_ratio)

# - Column 1: Average return
# Adjust standard errors for heteroscedasticity and autocorrelations
T = len(all_factors)
auto_lag = int(np.floor(4 * (T / 100)**(2/9)))

y = ['R_IA','R_ROE','HML','CMA','RMW_O','UMD']

Average_returns=[]
Average_returns_t=[]

# Run time series regressions and store results
for i in y:
    res=smf.ols(i + ' ~ 1', data=all_factors).fit(cov_type='HAC',cov_kwds={'maxlags':auto_lag})
    
    Average_returns.append(res.params[0])
    Average_returns_t.append(res.tvalues[0])

# - Column 2: 6-factor alpha
FF_6_factors=['MKT_RF','HML','SMB_O','CMA','RMW_O','UMD']
FF_6_factor_alphas=[]
FF_6_factor_alphas_t=[]

# Run time series regressions and store results
for o,i in enumerate(['R_IA','R_ROE']):
    res=smf.ols(i + ' ~ '+' + '.join(FF_6_factors), data=all_factors).fit(cov_type='HAC',cov_kwds={'maxlags':auto_lag})
    
    FF_6_factor_alphas.append(res.params[0])
    FF_6_factor_alphas_t.append(res.tvalues[0])

    # Store residuals for GRS test
    if o==0:
        FF_6_factor_residual=res.resid
    else:
        FF_6_factor_residual=pd.concat([FF_6_factor_residual,res.resid],axis=1)

# FF_6_factor_residual: T*2

# GRS-test
FF_alpha_GRS_pval=GRS_test(factor=all_factors[FF_6_factors].to_numpy(), resid=FF_6_factor_residual.to_numpy(), alpha=FF_6_factor_alphas)[1]
print('The 6-factor alphas of R_I/A and R_ROE=0: p={:.2f}'.format(GRS_test(factor=all_factors[FF_6_factors].to_numpy(), resid=FF_6_factor_residual.to_numpy(), alpha=FF_6_factor_alphas)[1]))

# - Column 3: q-factor alpha
q_factors_list=['R_MKT','R_ME','R_IA','R_ROE']
q_factor_alphas=[]
q_factor_alphas_t=[]

# Run time series regressions and store results
for o,i in enumerate(['HML','CMA','RMW_O','UMD']):
    res=smf.ols(i + ' ~ '+' + '.join(q_factors_list), data=all_factors).fit(cov_type='HAC',cov_kwds={'maxlags':auto_lag})
    
    q_factor_alphas.append(res.params[0])
    q_factor_alphas_t.append(res.tvalues[0])

    # Store residuals for GRS test
    if o==0:
        q_factor_residual=res.resid
    else:
        q_factor_residual=pd.concat([q_factor_residual,res.resid],axis=1)

# q_factor_residual: T*4

# GRS-test
q_alpha_GRS_pval=GRS_test(factor=all_factors[q_factors_list].to_numpy(), resid=q_factor_residual.to_numpy(), alpha=q_factor_alphas)[1]
print('The q-alphas of HML, CMA, RMW, and UMD=0: p={:.2f}'.format(GRS_test(factor=all_factors[q_factors_list].to_numpy(), resid=q_factor_residual.to_numpy(), alpha=q_factor_alphas)[1]))


# Replicate the table in LaTeX format
rows = [
    ("The investment factor, $R_{I/A}$", 0, 0, None),
    ("The Roe factor, $R_{\\text{Roe}}$", 1, 1, None),
    ("HML", 2, None, 0),
    ("CMA", 3, None, 1),
    ("RMW", 4, None, 2),
    ("UMD", 5, None, 3),
]

tex_lines = []
tex_lines.append("\\begin{table}[htbp]")
tex_lines.append("\\centering")
tex_lines.append("\\caption*{{\\small 1/1967--12/2024: The \\textit{{q}}-alphas of HML, CMA, RMW, and UMD = 0 ($p = {:.2f}$); the 6-factor alphas of $R_{{I/A}}$ and $R_{{\\text{{Roe}}}}$ = 0 ($p = {:.2f}$)}}".format(q_alpha_GRS_pval, FF_alpha_GRS_pval))
tex_lines.append("\\begin{tabular}{lccc}")
tex_lines.append("\\toprule")
tex_lines.append(" & \\textbf{Average} & \\textbf{6-factor} & \\textbf{q-factor} \\\\")
tex_lines.append(" & \\textbf{returns} & \\textbf{alphas} & \\textbf{alphas} \\\\")
tex_lines.append("\\midrule")

for name, avg_idx, ff6_idx, q_idx in rows:
    avg_val = f"{Average_returns[avg_idx]:.2f}"
    avg_t = f"({Average_returns_t[avg_idx]:.2f})"
    ff6_val = f"{FF_6_factor_alphas[ff6_idx]:.2f}" if ff6_idx is not None else ""
    ff6_t = f"({FF_6_factor_alphas_t[ff6_idx]:.2f})" if ff6_idx is not None else ""
    q_val = f"{q_factor_alphas[q_idx]:.2f}" if q_idx is not None else ""
    q_t = f"({q_factor_alphas_t[q_idx]:.2f})" if q_idx is not None else ""

    tex_lines.append(f"{name} & {avg_val} & {ff6_val} & {q_val} \\\\")
    tex_lines.append(f" & {avg_t} & {ff6_t} & {q_t} \\\\")

tex_lines.append("\\bottomrule")
tex_lines.append("\\end{tabular}")
tex_lines.append("\\end{table}")

# Write to .tex file
with open("latex_table/spanning_table.tex", "w") as f:
    for line in tex_lines:
        f.write(line + "\n")