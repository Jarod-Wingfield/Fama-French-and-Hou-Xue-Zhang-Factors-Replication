##########################################
# Replication of factors                 #
# Jiaying Wu                             #
# Date: April 2025                       #
# Updated: April 8th 2025                #
##########################################

import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *

import datetime
import os

import statsmodels.formula.api as smf
from linearmodels import FamaMacBeth 

import math
import statsmodels.api as sm
from collections import Counter


class Data_preprocess():
    def __init__(self,data_path, portfolio_month=6):
        self.data_path=data_path
        self.portfolio_month=portfolio_month

        pass

    ###################
    # Compustat Block #
    ###################

    # file_annually: Path to the Compustat Annual dataset
    # file_quarterly: Path to the Compustat Quarterly dataset
    def Compustat(self,file_annually,file_quarterly):

        comp=pd.read_csv(os.path.join(self.data_path,file_annually),low_memory=False)

        # https://www.crsp.org/products/documentation/annual-data-industrial
        comp=comp[['gvkey','cusip','datadate','datafmt','indfmt','sic','loc','popsrc','consol',
                   'at','pstkrv','pstkl','pstk','seq','txditc','revt','cogs','xsga','xint',
                   'rect','xpp','ap','invt','drc','drlt','xacc','ajex', 'csho','xrd']]

        # Variable descriptions (selected):
        # at: Total Assets
        # pstkl: Preferred Stock – Liquidation Value
        # pstkrv: Preferred Stock – Redemption Value
        # pstk: Preferred/Preference Stock – Total
        # seq: Stockholders’ Equity – Total
        # txditc: Deferred Taxes and Investment Tax Credit
        # revt: Total Revenue
        # cogs: Cost of Goods Sold
        # xsga: Selling, General & Administrative Expenses
        # xint: Interest Expense
        # rect, xpp, ap, invt, drc, drlt, xacc: Working capital components
        # ajex: Adjustment Factor
        # csho: Common Shares Outstanding

        # DATAFMT='STD' and CONSOL='C' and POPSRC='D' to retrieve the standardized (as opposed to re-stated data), 
        # consolidated (as opposed to pro-forma) data
        
        ##########################
        # Restrictions on data
        ##########################
        # In Fama French (1992), they exclude financial firms (indfmt==INDL).
        # Keep only standardized (non-restated), consolidated, domestic industrial-format data
        comp=comp[(comp['datafmt']=='STD')&(comp['consol']=='C')&(comp['popsrc']=='D')&(comp['indfmt']=='INDL')]
        comp['year']=pd.to_datetime(comp['datadate']).dt.year
        # Keep the latest one
        comp=comp.drop_duplicates(subset=['gvkey','year'],keep='last')
        comp=comp.sort_values(['gvkey','datadate'])

        # Keep the data with positive assets
        # comp=comp[comp['at']>0]

        # Filter by date range
        comp=comp[(pd.to_datetime(comp.datadate)>=datetime.datetime.strptime('19590101','%Y%m%d'))]
        # Def the time series order of stock
        comp['count']=comp.groupby(['gvkey']).cumcount()
        

        ##########################
        # Book Equity Calculation
        ##########################
        # Fama and French (1993): "I define book common equity, BE. as the COMPUSTAT book value of stockholders' equity, 
        # plus balance-sheet deferred taxes and investment tax credit (if available), minus the book value of preferred stock."

        # BE is the book value of stockholders' equity, plus balance sheet deferred taxes and investment tax credit (if available), minus the book value of preferred stock. 
        # Depending on availability, I use the redemption, liquidation, or par value (in that order) to estimate the book value of preferred stock. 
        
        # create preferrerd stock
        comp['ps'] = comp['pstkrv'].mask(comp['pstkrv'].isna(), comp['pstkl'])
        comp['ps'] = comp['ps'].mask(comp['ps'].isna(), comp['pstk'])

        # create book equity
        # set missing values of balance sheet deferred taxes and investment tax credits equal to zero.
        comp['be']=comp['seq']+comp['txditc'].fillna(0)-comp['ps']
        # comp['be']=np.where(comp['be']>0, comp['be'], np.nan)

        #############################################################
        # Sorting Variables for Fama-French 6-Factor Model
        #############################################################
        # Operating profitability (OP) = (Revenue – COGS – SG&A – Interest) / Book Equity
        # new of interest expense and scaled by book equity
        # revenues (revt) minus cost of goods sold (cogs), minus selling, general, and administrative expenses (xsga), minus interest expense (xint) all divided by book equity
        # fill nan to zero if at least one cogs, xsga, xint are non-missing
        comp.loc[comp[['cogs', 'xsga', 'xint']].notna().any(axis=1),['cogs', 'xsga', 'xint']]=\
            comp.loc[comp[['cogs', 'xsga', 'xint']].notna().any(axis=1),['cogs', 'xsga', 'xint']].fillna(0)

        comp['OP']=(comp['revt']-comp['cogs']-comp['xsga']-comp['xint']) / comp['be']
        
        # Cash profitability (CP) = (OP – Accruals) / Book Equity
        # accruals: the change in accounts receivable (rect) from t-2 to t-1, plus the change in prepaid expenses (xpp), minus the change in accounts payable (ap), inventory (invt), deferred revenue (drc+drlt), and accrued expenses (xacc)
        # According to Ball et al. (2016),  Instances where balance sheet accounts have missing values are replaced with zero values for the computation of cash-based operating profitability.
        comp['dr']=comp['drc']+comp['drlt']
        # change from t-2 to t-1
        comp[['delta_rect','delta_xpp','delta_ap','delta_invt','delta_dr','delta_xacc']]=comp[['gvkey','rect','xpp','ap','invt','dr','xacc']].groupby(['gvkey'],group_keys=False).apply(lambda x: x[['rect','xpp','ap','invt','dr','xacc']].apply(lambda y: y-y.shift(1)))
        # Ignore nan or replace nan to zero (Ball et al., 2016)
        comp['accruals']=comp[['delta_rect','delta_xpp']].sum(axis=1) - comp[['delta_ap','delta_invt','delta_dr','delta_xacc']].sum(axis=1)
        
        comp['CP']=((comp['revt']-comp['cogs']-comp['xsga']-comp['xint']) - comp['accruals']) / (comp['be'])
        
        # Investment (Inv) = (AT_t - AT_t-1) / AT_t-1
        # the change in total assets from t-2 to t-1 divided by total aseets at t-2.
        comp['Inv']=comp.groupby('gvkey',group_keys=False)['at'].apply(lambda x: (x-x.shift(1))/x.shift(1))

        #############################################################
        # Sorting Variables for Hou-Xue-Zhang q-Factor Model
        #############################################################
        # size [ME - market equity (stock price per share times shares outstanding from CRSP)], I/A, Roe
        
        # I/A: investment-to-assets, as the annual change in total assets (Compustat annual item AT) divided by one-year-lagged total assets.
        comp['ItA']=comp.groupby('gvkey',group_keys=False)['at'].apply(lambda x: (x-x.shift(1))/x.shift(1))
        
        # Roe: income before extraordinary items (Compustat quarterly item IBQ) divided by one-quarter-lagged book equity
        # the book equity is the quarterly version of the annual book equity
        # book equity is shareholders' equity, plus balance sheet deferred taxes and investment tax credit (item TXDITCQ) if available, minus the book value of preferred stock. 
        # stockholders' equity (item SEQQ) or common equity (item CEQQ) plus the carrying value of preferred stock (item PSTKQ), or total assets (item ATQ) minus total liabilities (item LTQ) in that order as shareholders' equity. use redemption value (item PSTKRQ) if available, or carrying value for the book value of preferred stock.
        
        # Load Quarterly Compustat
        comp_quarterly = pd.read_csv(os.path.join(self.data_path,file_quarterly),)
        comp_quarterly=comp_quarterly[(comp_quarterly['datafmt']=='STD')&(comp_quarterly['consol']=='C')&(comp_quarterly['popsrc']=='D')]
        comp_quarterly=comp_quarterly[['gvkey','rdq', 'datadate', 'fyearq', 'fqtr', 'fyr', 'indfmt', 'consol',
                                        'popsrc', 'datafmt', 'tic', 'cusip', 'conm', 'curcdq', 'datacqtr',
                                        'datafqtr', 'atq', 'ceqq', 'ibq', 'ltq', 'pstkq', 'pstkrq', 'seqq',
                                        'txditcq', 'exchg', 'cik', 'costat', 'fic', 'loc', 'sic','dvpsxq','cshoq','ajexq']]
        comp_quarterly=comp_quarterly.sort_values(['gvkey','datadate'])
        comp_quarterly=comp_quarterly.drop_duplicates()

        # Observe the duplicates by fiscal year-quarter
        comp_quarterly[comp_quarterly.duplicated(subset=['gvkey','datafqtr'],keep=False)]

        # # Keep the latest one
        # comp_quarterly=comp_quarterly.drop_duplicates(subset=['gvkey','datafqtr'],keep='last')

        # Calculate shareholders' equity hierarchy
        comp_quarterly['seq'] = comp_quarterly['seqq'].copy()
        comp_quarterly['seq'] = comp_quarterly['seq'].mask(comp_quarterly['seq'].isna(),comp_quarterly['ceqq'] + comp_quarterly['pstkq'])
        comp_quarterly['seq'] = comp_quarterly['seq'].mask(comp_quarterly['seq'].isna(),comp_quarterly['atq'] - comp_quarterly['ltq'])

        # the book value of preferred stock
        comp_quarterly['pstkrq'] = comp_quarterly['pstkrq'].mask(comp_quarterly['pstkrq'].isna(), comp_quarterly['pstkq'])

        # book equity
        comp_quarterly['beq'] = comp_quarterly['seq'] + comp_quarterly['txditcq'].fillna(0) - comp_quarterly['pstkrq']

        # Hou-Xe-Zhang (2025) "Technical Document" described the process to maximize the coverage for quarterly book equity
        # Supplement quarterly BE with annual BE for fiscal quarter 4 (Q4)
        # Convert datadate to datetime
        comp['datadate'] = pd.to_datetime(comp['datadate'])
        comp_quarterly['datadate'] = pd.to_datetime(comp_quarterly['datadate'])

        # Extract annual BE, CSHO, AJEX for merging
        annual_sup=comp[['gvkey', 'datadate', 'be', 'csho', 'ajex']]
        annual_sup.columns=['gvkey', 'datadate', 'annual_be', 'annual_csho', 'annual_ajex']

        # Merge annual BE, CSHO, AJEX into quarterly data where fqtr=4
        comp_quarterly=comp_quarterly.merge(annual_sup, on=['gvkey', 'datadate'], how='left')

        # Fill missing BE, CSHO, AJEX in Q4
        comp_quarterly['beq']=comp_quarterly['beq'].mask((comp_quarterly['fqtr']==4)&(comp_quarterly['beq'].isna()), comp_quarterly['annual_be'])
        comp_quarterly['cshoq']=comp_quarterly['cshoq'].mask((comp_quarterly['fqtr']==4)&(comp_quarterly['cshoq'].isna()), comp_quarterly['annual_csho'])
        comp_quarterly['ajexq']=comp_quarterly['ajexq'].mask((comp_quarterly['fqtr']==4)&(comp_quarterly['ajexq'].isna()), comp_quarterly['annual_ajex'])

        comp_quarterly = comp_quarterly.drop(columns=['annual_be', 'annual_csho', 'annual_ajex'])

        # Calculate dividends (DVQ) using available quarterly data
        comp_quarterly['dvq']=comp_quarterly['dvpsxq']*comp_quarterly['cshoq']/comp_quarterly['ajexq']

        # Sort by gvkey and datadate for imputation
        comp_quarterly = comp_quarterly.sort_values(['gvkey', 'datadate'])

        comp_quarterly=comp_quarterly.groupby('gvkey', group_keys=False).apply(impute_be)

        # As required in Hou-Xe-Zhang, I will exclude negative beq obs. (No this step in pre process)
        comp_quarterly['beq'] = np.where(comp_quarterly['beq']>0, comp_quarterly['beq'], np.nan)
        # Lag book equity by one quarter
        comp_quarterly['lag_beq'] = comp_quarterly.groupby('gvkey')['beq'].shift(1)
        
        # Calculate Roe = IBQ / lagged book equity
        comp_quarterly['Roe'] = comp_quarterly['ibq'] / comp_quarterly['lag_beq']

        # Final sort
        comp=comp.sort_values(by=['gvkey','datadate'])

        self.financial_var=['be','OP','CP','Inv','ItA','revt','cogs', 'xsga', 'xint']
        comp=comp[['gvkey','cusip','datadate','year','count','indfmt','sic','loc']+self.financial_var]
        
        self.comp=comp
        self.comp_quarterly=comp_quarterly
        return comp, comp_quarterly


    ###################
    # CRSP Block      #
    ###################
    # file_var: Path to the main CRSP monthly stock dataset
    # file_name: Path to CRSP header information (e.g., SHRCD, EXCHCD for identifying exchange and share types)
    # file_dl: Path to CRSP delisting returns data
    def CRSP(self,file_var,file_name,file_dl):
        # Load main CRSP monthly data
        crsp_m = pd.read_csv(os.path.join(self.data_path,file_var),low_memory=False)
        crsp_m=crsp_m[['PERMNO','PERMCO','TICKER','date','RET','RETX','SHROUT','PRC','NCUSIP','CUSIP']]
        crsp_m['date']=pd.to_datetime(crsp_m['date'])+MonthEnd(0)

        # permco	double	CRSP Permanent Company Number (permco)
        # ret	double	Holding Period Return (ret)
        # retx	double	Holding Period Return without Dividends (retx)
        # shrout	double	Number of Shares Outstanding (shrout)
        # prc	double	Price (prc)

        # Load CRSP header file for share and exchange codes
        crsp_name = pd.read_csv(os.path.join(self.data_path,file_name))
        crsp_name=crsp_name[['PERMNO','DATE','NAMEENDT','SHRCD','EXCHCD']]
        crsp_name=crsp_name[crsp_name.EXCHCD.isin([1,2,3])]
        crsp_name.columns=['PERMNO','NAMEDT','NAMEENDT','SHRCD','EXCHCD']

        # shrcd	double	Share Code (shrcd)
        # exchcd	double	Exchange Code (exchcd)
        # nameendt	date	Names Ending Date (nameendt)

        ##########################################
        # Filter and Merge Main and Header Files #
        ##########################################
        crsp_m1=pd.merge(crsp_m,crsp_name,on=['PERMNO'],how='left')
        crsp_m1=crsp_m1[(crsp_m1['date']>=crsp_m1['NAMEDT'])&(crsp_m1['date']<=crsp_m1['NAMEENDT'])]

        crsp_m1=crsp_m1[(pd.to_datetime(crsp_m1.date)>=datetime.datetime.strptime('19590101','%Y%m%d'))]

        # Map exchange codes to labels
        crsp_m1["exchange"] = crsp_m1["EXCHCD"].apply(assign_exchange)
        crsp_m1.exchange.value_counts()

        # Convert selected variables to integer type
        crsp_m1[['PERMCO','PERMNO','SHRCD','EXCHCD']]=crsp_m1[['PERMCO','PERMNO','SHRCD','EXCHCD']].astype('Int64')

        # Align dates to month-end
        crsp_m1['jdate']=pd.to_datetime(crsp_m1['date'])+MonthEnd(0)

        # Load and merge delisting returns
        dlret = pd.read_csv(os.path.join(self.data_path,file_dl))
        dlret=dlret[['PERMNO','DLRET','DLSTDT']]
        dlret.PERMNO=dlret.PERMNO.astype(int)
        
        # Line up date to be end of month
        dlret['jdate']=pd.to_datetime(dlret['DLSTDT'])+MonthEnd(0)

        crsp = pd.merge(crsp_m1, dlret, how='left',on=['PERMNO','jdate'])

        crsp['DLRET']=crsp['DLRET'].fillna(0)

        crsp['RET']=crsp['RET'].fillna(0)

        ##########################################
        # Clean Return                           #
        ##########################################
        # Convert return fields to numeric where needed
        crsp['RET']=crsp['RET'].apply(convert_currency)
        crsp['RETX']=crsp['RETX'].apply(convert_currency)

        # Handle invalid entries
        crsp['DLRET']=crsp['DLRET'].mask(crsp['DLRET'].apply(lambda x: isinstance(x,str)),0)
        crsp['RET']=crsp['RET'].mask(crsp['RET']=='C',0)
        crsp['RETX']=crsp['RETX'].mask(crsp['RETX']=='C',0)

        # Remove rows where RET is still not numeric
        crsp_now=crsp[(~crsp['RET'].apply(lambda x: isinstance(x,str)))]
        crsp_now = crsp_now.astype({"RET":'float', "RETX":'float'})

        # Adjust return to incorporate delisting return
        crsp_now['retadj']=(1+crsp_now['RET'])*(1+crsp_now['DLRET'])-1

        # Replace zero shares outstanding with NaN (to avoid zero market equity)
        crsp_now['SHROUT'].replace(0,np.nan,inplace=True)

        ##########################################
        # Compute Market Equity                  #
        ##########################################
        # calculate market equity
        crsp_now['me']=crsp_now['PRC'].abs()*crsp_now['SHROUT'] 
        crsp_now=crsp_now.drop(['DLRET','DLSTDT','SHROUT'], axis=1)
        crsp_now=crsp_now.sort_values(by=['jdate','PERMCO','me'])

        # Deal with the company which had not unique datas in a given date 
        # Notice: (If there are several values in one day of one particular permco stock, keep the one observation with maximum me.)

        ### Aggregate Market Cap —— For Value-Iighted ###
        # sum of me across different permno belonging to same permco a given date
        crsp_summe = crsp_now.groupby(['jdate','PERMCO'])['me'].sum().reset_index()
        # largest mktcap within a permco/date
        crsp_maxme = crsp_now.groupby(['jdate','PERMCO'])['me'].max().reset_index()
        # join by jdate/maxme to find the permno
        crsp1=pd.merge(crsp_now, crsp_maxme, how='inner', on=['jdate','PERMCO','me'])
        # drop me column and replace with the sum me
        crsp1=crsp1.drop(['me'], axis=1)
        # join with sum of me to get the correct market cap info
        crsp2=pd.merge(crsp1, crsp_summe, how='inner', on=['jdate','PERMCO'])
        # sort by permno and date and also drop duplicates
        # The same me observations of a given data and permco would keep and now be dropped.
        crsp2=crsp2.sort_values(by=['PERMNO','jdate']).drop_duplicates()

        # **Using pandas.sum(), the nan columns will return 0. Thus, I need to replace.**
        crsp2['me']=crsp2['me'].replace(0,np.nan)

        # keep December market cap : Use for computing its book-to-market, leverage, and earnings-price ratios
        crsp2['year']=crsp2['jdate'].dt.year
        crsp2['month']=crsp2['jdate'].dt.month
        decme=crsp2[crsp2['month']==12]
        decme=decme[['PERMNO','date','jdate','me','year']].rename(columns={'me':'dec_me'})

        ### Window_month set back self.portfolio_month month
        crsp2['ffdate']=crsp2['jdate']+MonthEnd(-self.portfolio_month)
        crsp2['ffyear']=crsp2['ffdate'].dt.year
        crsp2['ffmonth']=crsp2['ffdate'].dt.month

        crsp2['1+retx']=1+crsp2['RETX']
        crsp2=crsp2.sort_values(by=['PERMNO','date'])

        # cumret by stock
        crsp2['cumretx']=crsp2.groupby(['PERMNO','ffyear'])['1+retx'].cumprod()

        # lag cumret
        crsp2['lcumretx']=crsp2.groupby(['PERMNO'])['cumretx'].shift(1)

        # lag market cap
        crsp2['lme']=crsp2.groupby(['PERMNO'])['me'].shift(1)

        # if first permno then use me/(1+retx) to replace the missing value
        crsp2['count']=crsp2.groupby(['PERMNO']).cumcount()
        crsp2['lme']=np.where(crsp2['count']==0, crsp2['me']/crsp2['1+retx'], crsp2['lme'])

        # baseline me
        mebase=crsp2[crsp2['ffmonth']==1][['PERMNO','ffyear', 'lme']].rename(columns={'lme':'mebase'})

        # merge result back together
        crsp_monthly=pd.merge(crsp2, mebase, how='left', on=['PERMNO','ffyear'])

        # Calculate firm age
        crsp_monthly['ln_firm_age']=np.log(crsp_monthly.groupby(['PERMNO']).cumcount()+1)
        crsp_monthly['firm_age']=crsp_monthly.groupby(['PERMNO']).cumcount()+1

        # Notice: (Using the cumret is for adjusting market equity to the price level at base month(The first month of a portfolio))

        crsp_monthly['wt']=np.where(crsp_monthly['ffmonth']==1, crsp_monthly['lme'], crsp_monthly['mebase']*crsp_monthly['lcumretx'])

        crsp_monthly['retadj']=crsp_monthly['retadj']*100
        crsp_monthly['RET']=crsp_monthly['RET']*100

        crsp_monthly['date']=pd.to_datetime(crsp_monthly['date'])
        crsp_monthly['date']=crsp_monthly['date']+MonthEnd(0)

        self.crsp_monthly=crsp_monthly

        ##########################################
        # Construct Data at Portfolio Formation  #
        ##########################################
        # Shift fiscal ME to next year June
        decme['year']=decme['year']+1
        decme=decme[['PERMNO','year','dec_me']]

        # Info as of the Window Month. I usually construct portfolio in June for July to next year June.
        crsp_monthly_begin = crsp_monthly[crsp_monthly['month']==self.portfolio_month]

        crsp_month_win = pd.merge(crsp_monthly_begin, decme, how='inner', on=['PERMNO','year'])
        crsp_month_win=crsp_month_win[['PERMNO','TICKER','date','NCUSIP','CUSIP','jdate', 'SHRCD','EXCHCD','exchange','RET','PRC','retadj','me','wt','cumretx','mebase','lme','dec_me','ln_firm_age','firm_age']]
        crsp_month_win['ln_market_value']=np.log(crsp_month_win['me'])

        crsp_month_win=crsp_month_win.sort_values(by=['PERMNO','jdate']).drop_duplicates()

        self.crsp_month_win=crsp_month_win
    
        return crsp_monthly, crsp_month_win

    #######################
    # CCM Block           #
    #######################
    # file: path to the link table (Compustat and CRSP merge file)
    def CCM(self,file):
        ccm=pd.read_csv(os.path.join(self.data_path,file),low_memory=False)

        ccm=ccm[['gvkey','LPERMNO','LINKTYPE','LINKPRIM','LINKDT','LINKENDDT']]
        ccm.columns=['gvkey','PERMNO','LINKTYPE','LINKPRIM','linkdt','linkenddt']
        ccm['PERMNO']=ccm['PERMNO'].astype('Int64')

        # Keep only valid link types (L) and primary links (C or P)
        ccm=ccm[(ccm['LINKTYPE'].str[:1]=='L')&(ccm['LINKPRIM'].isin(['C','P']))]

        # If linkenddt is missing ('E'), replace with today's date
        ccm['linkenddt']=ccm['linkenddt'].mask(ccm['linkenddt']=='E',pd.to_datetime('today').strftime('%Y/%m/%d'))

        ccm1=pd.merge(self.comp,ccm,how='left',on=['gvkey'])
        ccm1['yearend']=pd.to_datetime(ccm1['datadate'])+YearEnd(0)

        # Important: shift the fiscal year end to June of the next year
        ccm1['jdate']=pd.to_datetime(ccm1['yearend'])+MonthEnd(self.portfolio_month)

        ccm1['linkdt']=pd.to_datetime(ccm1['linkdt'])
        ccm1['linkenddt']=pd.to_datetime(ccm1['linkenddt'])

        # Filter observations within the valid link date range
        ccm2=ccm1[(ccm1['jdate']>=ccm1['linkdt'])&(ccm1['jdate']<=ccm1['linkenddt'])]
        ccm2=ccm2[['gvkey','cusip','indfmt','PERMNO','datadate','yearend', 'jdate', 'count','sic']+self.financial_var]

        ccm2['year']=ccm2['jdate'].dt.year.copy()

        # Keep the most recent datadate for each PERMNO-jdate combination
        ccm2['datadate']=pd.to_datetime(ccm2['datadate'])

        ccm2=ccm2.sort_values(['PERMNO', 'jdate','datadate'],ascending=False)

        ccm2=ccm2.drop_duplicates(subset=['PERMNO', 'jdate'],keep='first')

        # Link annual Compustat data to CRSP using "jdate" (fiscal year end + 6 months)
        ccm_month_win=pd.merge(self.crsp_month_win, ccm2, how='inner', on=['PERMNO', 'jdate'])

        # Fama and French (2018): BE/ME is the ratio of book equity (Compustat) to market equity (CRSP, at Dec of t−1)
        ccm_month_win['beme']=ccm_month_win['be']*1000/ccm_month_win['dec_me']

        self.ccm_month_win=ccm_month_win

        self.financial_var=self.financial_var+['ln_firm_age','firm_age','ln_market_value']

        # Link quarterly Compustat data (for ROE) to CRSP monthly data
        ccm_quarterly=pd.merge(self.comp_quarterly,ccm,how='left',on=['gvkey'])
    
        ccm_quarterly['linkdt']=pd.to_datetime(ccm_quarterly['linkdt'])
        ccm_quarterly['linkenddt']=pd.to_datetime(ccm_quarterly['linkenddt'])

        # rdq: the most recent public quarterly earnings announcement dates.
        # I replace the nan by the final report date
        ccm_quarterly['datadate']=pd.to_datetime(ccm_quarterly['datadate'])
        ccm_quarterly['rdq']=pd.to_datetime(ccm_quarterly['rdq'])
        # ccm_quarterly['rdq']=ccm_quarterly['rdq'].mask(ccm_quarterly['rdq'].isna(),ccm_quarterly['datadate'])
        
        # Filter observations within the valid link date range using rdq (earnings announcement date)
        # print(ccm_quarterly.shape[0])
        ccm_quarterly=ccm_quarterly[(ccm_quarterly['rdq']>=ccm_quarterly['linkdt'])&(ccm_quarterly['rdq']<=ccm_quarterly['linkenddt'])]
        # print(ccm_quarterly.shape[0])

        ccm_quarterly=ccm_quarterly[['PERMNO','rdq','datadate','datafqtr','Roe']]

        self.crsp_monthly['begin_of_month']=pd.to_datetime(self.crsp_monthly['date'])+pd.offsets.MonthBegin(-1)

        # Merge ROE quarterly data to CRSP monthly data
        # Step 1: Find the closest rdq <= beginning of month
        self.crsp_monthly=pd.merge_asof(
            self.crsp_monthly.sort_values('begin_of_month'),
            ccm_quarterly[["PERMNO", 'rdq','datafqtr', "Roe"]].sort_values('rdq'),
            left_on="begin_of_month",
            right_on="rdq",
            by="PERMNO",
            direction="backward",  # Latest rdq <= month_end
        )

        # Step 2: Keep only if rdq is within six months of the quarter-end
        # fiscal quarter to month end
        self.crsp_monthly['fiscal_quarter_month_end']=pd.Series(pd.PeriodIndex(self.crsp_monthly['datafqtr'], freq='Q').to_timestamp(how='end')).dt.to_period('M')
        self.crsp_monthly['begin_of_month']=self.crsp_monthly['begin_of_month'].dt.to_period('M')
        # Convert periods to total month counts for comparison
        self.crsp_monthly['begin_of_month'] = self.crsp_monthly['begin_of_month'].dt.year * 12 + self.crsp_monthly['begin_of_month'].dt.month
        self.crsp_monthly['fiscal_quarter_month_end'] = self.crsp_monthly['fiscal_quarter_month_end'].mask(self.crsp_monthly['fiscal_quarter_month_end'].notna(), self.crsp_monthly['fiscal_quarter_month_end'].dt.year * 12 + self.crsp_monthly['fiscal_quarter_month_end'].dt.month)

        # Replace the Roe to nan if outside six month window
        self.crsp_monthly['Roe']=self.crsp_monthly['Roe'].mask(self.crsp_monthly['begin_of_month']-1-self.crsp_monthly['fiscal_quarter_month_end']>6, np.nan)

        return ccm_month_win, self.crsp_monthly
    

def assign_exchange(exchcd):
    if exchcd in [1]:
        return "NYSE"
    elif exchcd in [2]:
        return "AMEX"
    elif exchcd in [3]:
        return "NASDAQ"
    else:
        return "Other"

def convert_currency(var):
    try:
        var = float(var)
    except:
        var = var
    return var


# Impute missing BEQ using clean surplus relation (backward looking up to 4 quarters)
def impute_be(group):
    for idx in group.index:
        if pd.isna(group.at[idx, 'beq']):
            # Look back up to 4 quarters
            for lag in range(1, 5):
                prev_idx = idx - lag
                if prev_idx in group.index:
                    prev_be = group.at[prev_idx, 'beq']
                    if not pd.isna(prev_be):
                        # Sum IBQ and DVQ betIen prev_idx+1 and current idx
                        subset = group.loc[prev_idx+1:idx]
                        sum_ibq = subset['ibq'].sum()
                        sum_dvq = subset['dvq'].sum()
                        group.at[idx, 'beq'] = prev_be + sum_ibq - sum_dvq
                        break
    return group