#preprocessing script to binned and imputed final data to apply on simple baselines..

import pandas as pd
import numpy as np 
import sys

def bin_and_impute(data, bin_width=60, variable_start_index=5):
    result = [] #list of patients dataframes

    #set of variables to process:
    variables = np.array(list(data.iloc[:,variable_start_index:]))
    #create resample parameter string:
    bin_width = str(bin_width)+'min'

    #all distinct icustay ids:
    id_s = data['icustay_id'].unique() # get unique ids

    #loop over patients:
    for icustay_id in id_s:
        print(f'Processing ID: {icustay_id} ....')
        pat = data.query( "icustay_id == @icustay_id" ) #select subset of dataframe featuring current icustay_id
        pat_i = pat.set_index('chart_time', inplace=False)     

        #resampling needs datetime or timedelta format, create dummy timestamp from relative hour:
        #pat_i.index = pd.to_datetime(pat_i.index, unit='D') #unit='s'
        pat_i.index = pd.to_timedelta(pat_i.index, unit='h')

        # if first index > 0, add empty point to start with (such that reampling occurs on same grid as TCN/RNN grid )
        start = pd.to_timedelta(0.0, unit='h')
        if pat_i.index[0] > start:
            #take the first row and set the variables to nan, and the index to 0. append first row with pat_i
            first_row = pat_i.iloc[0].copy(deep=True)
            first_row[variables] = np.nan
            first_row = pd.DataFrame(first_row, columns=pat_i.columns, index=[start])
            pat_a = first_row.append(pat_i)
        else:
            pat_a = pat_i.copy(deep=True)
        
        #Patient with no measurements in extracted window can not be resampled -> will be removed later on (when masking_samples)
        #for here, simply replace all vars with 0, as they will be dropped when masking
        n_non_nans = (~pat[variables].isnull()).sum().sum()
        if n_non_nans == 0:
            pat_a[variables] = pat[variables].replace(np.nan, 0)

        #resampling to bins of bin_width size:
        pat_rs = pat_a[variables].resample(bin_width, how='mean') #loffset=pat_i.index[0], label='left'
        
        #forward filling
        pat_ff = pat_rs.ffill()
        #fill remaining NaN with train mean (sample and if sample mean is nan -> train_stats mean!)
        for variable in variables:
            n_nans = pat_ff[variable].isnull().sum()
            if n_nans > 0: #if yes, this variable/channel has still NaNs --> address with mean imputation
                if n_nans == len(pat_ff[variable]):
                    #do zero-imputation (as data is already centered)
                    replacement = 0 # as dataset mean is 0 after centering
                else:
                    #sample mean imputation
                    replacement = pat_ff[variable].mean()

                pat_ff[variable] = pat_ff[variable].replace(np.nan, replacement)
        if pat_ff.isnull().sum().sum() > 0:
            print(f'NaN REMAINING for patient {icustay_id} !!')
        
        result.append(pat_ff)
    return result, id_s
