
'''
---------------------------------
Author: Michael Moor, 11.10.2018
---------------------------------
This code is used to preprocess vital/lab data in a way that the format can be directly used for mgp-rnn baseline. Our proposed methods start out from this format too.

'''

import argparse
import pandas as pd
from pandas import Series
import numpy as np

from datetime import datetime

import csv
import json

import sys #for exiting the script
import time

import os.path # check if file exists
import pickle

#import preprocessing scripts:
from .collect_records import *
from .bin_and_impute import bin_and_impute

#Function to extract time window for cases and controls!
def extract_window(data=None, static_data=None, onset_name=None, horizon=0):
    result = pd.DataFrame()
    ids = data['icustay_id'].unique()
    for icuid in ids:
        pat = data.query( "icustay_id == @icuid" ) # select only rows where icustay_id matches 
        pat = pat.set_index(pd.DatetimeIndex(pat['chart_time'])) # set chart_time as index, such that window extraction works
        
        start = static_data[static_data['icustay_id']==icuid]['intime'].values[0] #determine start time (icu-intime)
        end = static_data[static_data['icustay_id']==icuid][onset_name].values[0] #determine end time (onset of sepsis or control onset)
        early_end = end - pd.Timedelta(hours=horizon) # define earlier end of extraction window depending on prediction horizon! (in hours)
        
        pat_window = pat[start:early_end] #select window from in-time up to onset minus horizon padding.
        pat_window['chart_time'] = (pat_window['chart_time']-start)/pd.Timedelta(hours=1) # convert chart_time to relative hour
        result=result.append(pat_window)
    return result

#Function to standardize X based on X_train:
def standardize(train=None, val=None, test=None, variable_start_index=5):
    variables = np.array(list(test.iloc[:,variable_start_index:]))
    
    #initialize header info:
    train_z = train.copy(deep=True)
    val_z = val.copy(deep=True)
    test_z = test.copy(deep=True)
    
    #get train stats (check if non-nan!)
    mean = train[variables].mean(axis=0)
    std = (train[variables]-mean).std(axis=0)
    if (mean.isnull().sum() or std.isnull().sum()) > 0:
        print('Nan in statistics found.. increase na_thres of droping columns')

    #standardize by train statistics:
    train_z[variables] = (train[variables]-mean)/std
    val_z[variables] = (val[variables]-mean)/ std 
    test_z[variables] = (test[variables]-mean)/ std  

    stats = {'mean': np.array(mean).tolist(), 'std': np.array(std).tolist()}

    return train_z, val_z, test_z, stats   


def select_static_vars(df, static_vars): # returns a selection of used static variables with simplified categories.
    start_time = time.time()
    #convert age to bins (>70, >50, <50)
    #convert ethn to summary vars
    df = df[static_vars] # first select list of variables for further processing
        
    df_out = df.copy()
    df_out = df_out.drop(columns=['ethnicity','admission_age'])
    df_out['ethnicity'] = np.nan
    df_out['admission_age'] = np.nan
    df_out = df_out[static_vars]

    #Line-by-Line processing (faster and looping over entire df multiple times..)
    for row in df_out[:10].itertuples(): #start with 10 rows for debugging
        row_ind = row[0]
        old_row = df.iloc[row_ind]
        #Loop over columns, replace with simplified / discretized covariate variables
        for col_ind, item in zip(old_row.index, old_row):
            #process ethnicity column:
            if col_ind == 'ethnicity':
                if item in ['BLACK/AFRICAN AMERICAN','BLACK/CAPE VERDEAN']:
                    new_eth = 'black'
                elif item == 'WHITE':
                    new_eth = 'white'
                elif item in ['UNKNOWN/NOT SPECIFIED','UNABLE TO OBTAIN']:
                    new_eth = 'na'
                else:
                    new_eth = 'other'
                df_out.loc[row_ind, col_ind] = new_eth
            #process if age column:
            if col_ind == 'admission_age':
                if item > 70:
                    new_age = '>70'
                elif item > 50:
                    new_age = '>50'
                elif item <= 50:
                    new_age = '<=50'
                df_out.loc[row_ind, col_ind] = new_age
    
    dummy_vars = static_vars[1:]
    df_dummy = pd.get_dummies(df_out[dummy_vars]) #create one-hot dummy variables for all static variables used for the model (not icustay_id)
    df_dummy.insert(loc=0, column='icustay_id', value=df_out['icustay_id'].values)
    print('Select static variables took {}'.format(time.time() - start_time))

    return df_dummy

#function to drop samples with too short time series (define by min_length)
def drop_short_series_old(data, min_length=7):
    result = pd.DataFrame()
    ids = data['icustay_id'].unique()
    for icuid in ids:
        #process current icustay_id
        pat = data.query( "icustay_id == @icuid" ) #select only those rows where icustay_id matches current icuid iteration
        # Get end_time for num-/grid_times:
        end_time = pat['chart_time'].iloc[-1] # last used time of this icustay
        # Determine size on grid:
        num_rnn_grid_time = int(np.floor(end_time)+1)
        ##if too short, remove this sample:
        if num_rnn_grid_time < min_length:
            #print('Skipping patient {}'.format(icuid))
            continue
        else: 
            result=result.append(pat)
    return result

def drop_short_series(data, case_static, control_static, min_length=7, max_length=200):
    #Determine if onset hour earlier than min_length:
    long_cases = case_static[case_static['sepsis_onset_hour']>=min_length]['icustay_id'].values
    long_controls= control_static[control_static['control_onset_hour'] >=min_length]['icustay_id'].values
    selected_patients = np.concatenate([long_cases, long_controls])
    #intialize result
    result = pd.DataFrame()
    ids = data['icustay_id'].unique()

    cases, controls = 0,0
    for icuid in ids:
        #process current icustay_id
        pat = data.query( "icustay_id == @icuid" ) #select only those rows where icustay_id matches current icuid iteration
        # Get end_time for num-/grid_times:
        end_time = pat['chart_time'].iloc[-1] # last used time of this icustay
        # Determine size on grid:
        num_rnn_grid_time = int(np.floor(end_time)+1)
        
        if icuid not in selected_patients: #if onset earlier than min_length, continue to next patient   
            #print('Skipping patient {}, TS too short'.format(icuid))
            continue
        ##if too long, skip this outlier sample and continue to next patient; (those 5 outlier patients (>200) almost quadruple memory usage..)
        elif num_rnn_grid_time > max_length:
            #print('Skipping patient {}, TS too long'.format(icuid))
            continue
        else: #if both exclusion criteria do not apply, add patient to resulting df
            result=result.append(pat)
            if icuid in long_cases:
                cases+=1 
            elif icuid in long_controls:
                controls+=1
    print('Using {} cases, {} controls.'.format(cases, controls))
    return result

def get_onset_hour(case_static, control_static):
    result = pd.DataFrame(columns=['icustay_id', 'onset_hour']) # df to return mapping icustay_id to onset_hour (true sepsis onset or matched control onset)
    #prepare case and control info such that they can be joined into same df:
    case_hour = case_static[['icustay_id', 'sepsis_onset_hour']]
    control_hour = control_static[['icustay_id', 'control_onset_hour']]
    for dataset in [case_hour, control_hour]:
        new_cols = {x: y for x, y in zip(dataset.columns, result.columns)}
        dataset = dataset.rename(columns=new_cols)
        result = result.append(dataset)
    return result


#Function to transform data from sparse df format to compact format as to feed to mgp rnn script.
def compact_transform(data, onset_hours, variable_start_index=5):
    start_time = time.time()
    #initialize outputs:
    values = [] # values[i][j] stores lab/vital value of patient i as jth record (all lab,vital variable types in same array!) 
    times = [] # times[i][:] stores all distinct record times of pat i (hours since icu-intime) (sorted)
    ind_lvs = [] # ind_lvs[i][j] stores index (actually id: e.g. 0-9) to determine the lab/vital type of values[i][j]. 
    ind_times = [] # ind_times[i][j] stores index for times[i][index], to get observation time of values[i][j].
    labels = [] # binary label if case/control
    num_rnn_grid_times = [] # array with numb of grid_times for each patient
    rnn_grid_times = [] # array with explicit grid points: np.arange of num_rnn_grid_times
    num_obs_times = [] #number of times observed for each encounter; how long each row of T (times) really is
    num_obs_values = [] #number of lab values observed per encounter; how long each row of Y (values) really is
    onset_hour = [] # hour, when sepsis (or control onset) occurs (practical for horizon analysis)

    #Process all patients:    
    ids = data['icustay_id'].unique()
    for icuid in ids: 
        #process current icustay_id
        pat = data.query( "icustay_id == @icuid" ) #select only those rows where icustay_id matches current icuid iteration
    
        # Get end_time for num-/grid_times:
        end_time = pat['chart_time'].iloc[-1] # last used time of this icustay
        # Determine size on grid:
        num_rnn_grid_time = int(np.floor(end_time)+1)
        
        num_rnn_grid_times.append(num_rnn_grid_time)
        rnn_grid_times.append(np.arange(num_rnn_grid_time))

        # Get measurement observation times:
        pat_times = pat['chart_time'].values
        times.append(pat_times)
        num_obs_times.append(len(pat_times))

        # Write label to labels
        labels.append(pat['label'].values[0])
        
        # Loop over variables to get values:
        variables = np.array(list(pat.iloc[:,variable_start_index:])) #get variable names

        pat_values = [] #values list of current patient
        pat_ind_lvs = [] #ind_lvs list of curr patient
        pat_ind_times = [] #ind_times list of curr patient
        #WHAT ELSE DO WE NEED?
        for v_id, var in enumerate(variables): #loop over variable ids and names
            vals = pat[[var, 'chart_time']].dropna().values # values and corresponding time which are non-nan
            for val, chart_time in vals:
                pat_values.append(val)
                pat_ind_lvs.append(v_id)
                time_index = np.where(pat_times == chart_time)[0][0] # get index of pat_times where time matches
                pat_ind_times.append(time_index) # append index with which pat_times[index] return chart time of current value
        values.append(np.array(pat_values)) #append current patients' values to the overall values list
        num_obs_values.append(len(pat_values)) #append current patients' number of measurements
        ind_lvs.append(np.array(pat_ind_lvs)) #append current patients' indices of labvital ids to overall ind_lvs list
        ind_times.append(np.array(pat_ind_times)) #append current patients' indices of times to overall ind_times list
        
        #Append onset hour of current patient:
        onset_hour.append(onset_hours.loc[onset_hours['icustay_id']==icuid]['onset_hour'].values[0])

    results = [np.array(item) for item in [values,times,ind_lvs,ind_times,labels,num_rnn_grid_times,rnn_grid_times,num_obs_times,num_obs_values,onset_hour]]
    print('Reformatting to compact format took {} seconds'.format(time.time() - start_time)) # reformatting test data took 170s, not really worth parallelizing..
    return results

#Main function:
def load_data(test_size=0.1, horizon=0, na_thres=500, variable_start_index=5, data_sources=['labs','vitals','covs'], min_length=None, max_length=None, overwrite=False, split=0, binned=False):
    #TODO: make vitals and labs independent (such that not both are necessary..)

    #Parameters:
    
    rs = np.random.RandomState(split)
    
    #---------------------------------
    # 0. SET PATHS (hard-coded relative paths for now)
    # outpath to case/control-joined and window extracted file
    labvital_outpath='output/full_labvitals_horizon_{}.csv'.format(horizon)
    #Case vitals and labs (input, output)
    case_vitals_in='output/case_55h_hourly_vitals_ex1c.csv'
    case_vitals_out='output/case_55h_hourly_vitals_ex1c_collected.csv'
    case_labs_in='output/case_55h_hourly_labs_ex1c.csv'
    case_labs_out='output/case_55h_hourly_labs_ex1c_collected.csv'
    #Control vitals and labs (input, output)
    control_vitals_in = 'output/control_55h_hourly_vitals_ex1c.csv'
    control_vitals_out = 'output/control_55h_hourly_vitals_ex1c_collected.csv'
    control_labs_in = 'output/control_55h_hourly_labs_ex1c.csv'
    control_labs_out = 'output/control_55h_hourly_labs_ex1c_collected.csv'

    #Time Series output (splitted)
    compact_split_out = 'output/labvitals_tr_te_val_compact_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(min_length,max_length, horizon,split)
    binned_split_out = 'output/labvitals_tr_te_val_binned_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(min_length,max_length, horizon,split)

    #Static data:
    case_static_in='output/static_variables_cases.csv'
    control_static_in='output/static_variables_controls.csv'

    #Load static info (with onset hour / times)
    case_static = pd.read_csv(case_static_in)
    for t in ['intime','sepsis_onset']: #convert string (of times) to datetime objects
        case_static[t] = case_static[t].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )

    control_static = pd.read_csv(control_static_in)
    for t in ['intime','control_onset_time']: 
        control_static[t] = control_static[t].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )
    
    # Align onset hours of cases and controls in one simple df to map id to onset_hour (for compact_transform)
    onset_hours = get_onset_hour(case_static, control_static)

    if overwrite or not os.path.isfile(labvital_outpath):
        # IF NOT ALREADY RUN, UNCOMMENT THIS SECTION:
        if overwrite or not os.path.isfile(case_vitals_out): #if multi-row records data was not collected in single row per observation time before, do it now
            
            #---------------------------------
            # 1. a): Collect Data line-by-line
            print('First run of this setting. Collecting Data...')
            # First collect the data of the sql-queried csv files (case and control vitals) in single rows for each patient & point in time:

            print('Collecting Case vitals...')
            collect_records(case_vitals_in,case_vitals_out)
            #collect_records(args.infile_cases, args.outfile_collected_cases)

            print('Collecting Case labs...')
            collect_records(case_labs_in,case_labs_out)

            print('Collecting Control vitals...')

            collect_records(control_vitals_in,control_vitals_out)

            print('Collecting Control labs...')

            collect_records(control_labs_in,control_labs_out)

            #collect_records(args.infile_controls, args.outfile_collected_controls)
            #uncomment this condition, to not overwrite:
        else:
            print('This data has been collected before, load from csv..')
            
        #---------------------------------
        # 1. b): Load collected data

        print('Loading collected case records...')
        #Read cases:
        case_vitals = pd.read_csv(case_vitals_out) #read file to pd.dataframe
        case_vitals['chart_time'] = case_vitals['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") ) # convert time string to datetime object
        case_labs = pd.read_csv(case_labs_out)
        case_labs['chart_time'] = case_labs['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )

        print('Loading collected control records...')
        #Read controls:
        control_vitals = pd.read_csv(control_vitals_out) #read file to pd.dataframe
        control_vitals['chart_time'] = control_vitals['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") ) # convert time string to datetime object
        control_labs = pd.read_csv(control_labs_out)
        control_labs = control_labs.dropna(subset=['chart_time']) # had to include this dropna row as few severly incomplete records in labevents table
        control_labs = control_labs.reset_index(drop=True) # re-enumerate the row index after removing few noisy rows missing chart_time (173)
        control_labs['chart_time'] = control_labs['chart_time'].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )


        #---------------------------------
        # 2. a): Merge lab with vital time series (from different sql tables originally) and append case and controls to one dataframe!
        print('Merge lab and vital time series data ..')
        #CASE: Merge vital and lab values into one time series:
        case_labvitals = pd.merge(case_vitals, case_labs, how='outer', left_on=['icustay_id', 'chart_time', 'subject_id', 'sepsis_target'], 
            right_on=['icustay_id', 'chart_time', 'subject_id', 'sepsis_target'], sort=True)

        #CONTROL: Merge vital and lab values into one time series:
        control_labvitals = pd.merge(control_vitals, control_labs, how='outer', left_on=['icustay_id', 'chart_time', 'subject_id', 'pseudo_target'], 
            right_on=['icustay_id', 'chart_time', 'subject_id', 'pseudo_target'], sort=True)

        #---------------------------------
        # 2. b): Extract case and control window time series
        print('Extract time series window 48 hour before onset for prediction') 
        case_labvitals = extract_window(data=case_labvitals, static_data=case_static, onset_name='sepsis_onset', horizon=horizon)
        control_labvitals = extract_window(data=control_labvitals, static_data=control_static, onset_name='control_onset_time', horizon=horizon)
        # in the extract_window() step we drop 633 control stays from 17909 control stays to 17276, as for some controls there is no data in this window (luckily no losses on cases!)

        #---------------------------------
        # 2. c): Join Cases and Controls
        print('Merge case and control data')
        #rename pseudo_target, such that case and controls can be appended to same df..
        control_labvitals = control_labvitals.rename(columns={'pseudo_target': 'sepsis_target'})
        #for joining label cases/controls with label: 1/0
        control_labvitals.insert(loc=0, column='label', value=np.repeat(0,len(control_labvitals)))
        case_labvitals.insert(loc=0, column='label', value=np.repeat(1,len(case_labvitals)))
        #append cases and controls, for spliting/standardizing:
        full_labvitals = case_labvitals.append(control_labvitals)
        full_labvitals=full_labvitals.reset_index(drop=True) #drop chart_time index, so that on-the-fly df is identical with loaded one
        full_labvitals.to_csv(labvital_outpath, index=False)

    else:
        print('full_labvitals_horizon_{}.csv exists, cases/control were merged before and window extracted. Load this file..'.format(horizon))

    full_labvitals = pd.read_csv(labvital_outpath)
    full_labvitals = full_labvitals.dropna(axis=1, thresh=na_thres) # drop variables that don't at least have na_thres many measurements..
    #drop too short time series samples.
    if min_length or max_length:
        #handle edge cases to prevent unexpected bugs
        if min_length is None:
            min_length=0
        if max_length is None:
            max_length=100000
        full_labvitals = drop_short_series(full_labvitals, case_static, control_static, min_length=min_length, max_length=max_length)

    #---------------------------------
    # 2. d) Process static raw variables
    #---------------------------------
    if 'covs' in data_sources: # if covs required: process them
        static_vars = ['icustay_id','gender','admission_age','ethnicity','first_careunit']
        #Create summary columns:
        print('Process CASE static variables')
        case_static_sum = select_static_vars(case_static, static_vars)    
        print('Process CONTROL static variables')
        control_static_sum = select_static_vars(control_static, static_vars) #cave: still contains more ids than subsequently, as not for all patients data available during extraction window
        #Make sure that static cases and controls have identical columns:
        if (control_static_sum.columns != case_static_sum.columns).sum() > 0:
            raise ValueError('static case/control columns could not be aligned. Check that both actually have all selected_vars')
        full_static = case_static_sum.append(control_static_sum)

    #---------------------------------
    # 3. a): Create Splits 
    #---------------------------------
    print('Creating Splits..')

    #Get list of actual ids (for which there was data during extraction window)
    all_ids = full_labvitals['icustay_id'].unique() #get icustay_ids of all patients
    #case_ids = case_labvitals['icustay_id'].unique()
    case_ids = full_labvitals[full_labvitals['label']==1]['icustay_id'].unique() # use full_labvitals for case_ids as case_labvitals might no be available if second run 
    #control_ids = control_labvitals['icustay_id'].unique()
    control_ids = full_labvitals[full_labvitals['label']==0]['icustay_id'].unique()

    #Createt train/test/val split ids:
    tvt_perm = rs.permutation(len(all_ids)) #train/val/test permutation of indices
    split_size = int(test_size*len(all_ids))
    test_ind = tvt_perm[:split_size]
    test_ids= all_ids[test_ind]
    validation_ind = tvt_perm[split_size:2*split_size]
    validation_ids = all_ids[validation_ind]
    train_ind = tvt_perm[2*split_size:]
    train_ids = all_ids[train_ind]
    #write TVTsplit ids of current split to json
    tvt_ids = {'train': train_ids, 'validation': validation_ids, 'test': test_ids}
    pickle.dump(tvt_ids, open(f'output/tvt_info_split_{split}.pkl', 'wb') )

    ##sanity check that split was reasonably balanced (not by chance no cases in test split)
    test_prev = len(set(test_ids).intersection(case_ids))/len(test_ids) # is it comparable to overall prevalence of 9%?
    #print('Splitting: random perm of ids: {} ... '.format(tvt_perm[:10]))

    print('Split ids set up!')
    #Currently, always process lab/vital timeseries
    #actually split timeseries data: (still labeled) 
    print('Split test time series!')
    test_data = full_labvitals[full_labvitals['icustay_id'].isin(test_ids)]
    print('Split train time series!')
    train_data = full_labvitals[full_labvitals['icustay_id'].isin(train_ids)]
    print('Split validation time series!') 
    validation_data = full_labvitals[full_labvitals['icustay_id'].isin(validation_ids)]

    if 'covs' in data_sources:
        #Create splits for static data:
        print('Split test statics!') 
        test_static_data = full_static[full_static['icustay_id'].isin(test_ids)]
        print('Split train statics!') 
        train_static_data = full_static[full_static['icustay_id'].isin(train_ids)]
        print('Split val statics!') 
        validation_static_data = full_static[full_static['icustay_id'].isin(validation_ids)]
        #After splitting drop id column and convert to np array as for mgp_rnn_fit script
        test_static_data= np.array( test_static_data.drop(columns=['icustay_id']) )
        train_static_data = np.array( train_static_data.drop(columns=['icustay_id']) )
        validation_static_data = np.array( validation_static_data.drop(columns=['icustay_id']) )

    #---------------------------------
    # 3. b): and Standardize (still in df format, not compact one)
    print('Standardizing time series!')
    train_z,val_z,test_z, stats = standardize(train=train_data, val=validation_data, test=test_data, variable_start_index=variable_start_index)
    variables = np.array(list(train_z.iloc[:,variable_start_index:]))
    
    # Write used stats and variable names to json for easier later processing (temporal signature creation)
    stats['names'] = variables.tolist()
    with open(f'output/temporal_signature_info_split_{split}.json', 'w') as jf:
        json.dump(stats, jf)

    #If data should be binned (for simple baselines) call additional prepro script, otherwise transform irregular data to compact format for mgp-tcn/mgp-rnn methods
    if binned:
        if overwrite or not os.path.isfile(binned_split_out): 
            #call prepro script to bin and impute (carry forward) data for simple baselines
            print('Binned time series not dumped yet, computing and dumping it...')
            #TODO: call prepro script on each split (returning train_data, ..)
            train_data, sorted_train_ids = bin_and_impute(data=train_z, variable_start_index=variable_start_index)
            validation_data, sorted_validation_ids = bin_and_impute(data=val_z, variable_start_index=variable_start_index)
            test_data, sorted_test_ids = bin_and_impute(data=test_z, variable_start_index=variable_start_index)
            
            datadump = [variables, train_data, validation_data, test_data]
            pickle.dump( datadump, open(binned_split_out, "wb"))
            #dump also sorted ids for easier processing with DTW matrix..
            sorted_ids = {'train': sorted_train_ids, 'validation': sorted_validation_ids, 'test': sorted_test_ids}
            pickle.dump(sorted_ids, open(f'output/tvt_sorted_info_split_{split}.pkl', 'wb') )
            #with open(f'../../output/tvt_sorted_info_split_{split}.json', 'w') as sf:
            #    json.dump(sorted_ids, sf)

        else: #read previously dumped data
            print('Binned time series available, loading it..')
            variables, train_data, validation_data, test_data = pickle.load( open( binned_split_out, "rb" ))
        #Return the data (not the static here, it should be loaded from compact format as there the masking is applied)
        return [variables, train_data, validation_data, test_data]

    else:
        #---------------------------------
        # 4. Transform Data into compact format for feeding to MGP-RNN 
        print('Compact transform of time series..!')

        # Dump this step as it takes long for quicker prototyping:
        if overwrite or not os.path.isfile(compact_split_out): 
            print('Compact transformed time series not dumped yet, computing and dumping it..')
            train_data = compact_transform(data=train_z, onset_hours=onset_hours, variable_start_index=variable_start_index)
            validation_data = compact_transform(data=val_z, onset_hours=onset_hours, variable_start_index=variable_start_index)
            test_data = compact_transform(data=test_z, onset_hours=onset_hours, variable_start_index=variable_start_index)
            datadump = [variables, train_data, validation_data, test_data]
            pickle.dump( datadump, open(compact_split_out, "wb"))
        else: #read previously dumped data
            print('Compact transformed time series available, loading it..')
            variables, train_data, validation_data, test_data = pickle.load( open( compact_split_out, "rb" ))

        #Return the data:
        #for more flexible handling, return a variable-length list of data sources (currently only handling case: labvital vs labvital+cov)
        if 'covs' in data_sources:
            return [variables, train_data, validation_data, test_data, train_static_data, validation_static_data, test_static_data]
        else:
            return [variables, train_data, validation_data, test_data]






