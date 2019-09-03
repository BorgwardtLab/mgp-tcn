'''
---------------------------------
Author: Michael Moor, 13.03.2018
---------------------------------

Collect records that are spread over different rows in input file such that
    1) records of same icustay_id and chart_time are on same row
    2) redundant values for one id and time are averaged over

This script is able to process a varying number of medical variables assuming that they are listed in columns after (or instead of) the other three variables.

collect_records(file_in_path, file_out_path):
    input: path to file to be processed
    output: processed file at file_out_path   
'''

import numpy as np
import pandas as pd
import sys
import csv
import time


def collect_records(filepath=None, outpath=None):

    print("Collecting extracted database records in single lines per observation time")
    start = time.time() # Get current time

    # -----------------------------------------------------------------------------
    #Step 1. Open the input file:
    try:
        f_in = open(filepath, 'r')
    except IOError:
        print("Cannot read input file %s" % filepath)
        sys.exit(1)


    # -----------------------------------------------------------------------------
    #Step 2. read file line by line

    #first read header by reading first line
    line = f_in.readline()

    #print('First Line: {}'.format(line))

    parts = line.rstrip().split(",") # split the line

    firstrow2write = parts # will be written to the output file as first line

    #print('First parts: {}'.format(parts))

    variable_indices = range(4, len(parts))
    variables = parts[4: len(parts)] #containing the medical variable names
    var_counter = np.zeros(len(variable_indices)) #counting how many observation for certain timepoint available (for averaging over)
    var_sum = np.repeat(np.nan, len(variable_indices)) #summing the value of all variables for certain timepoint

    #process first line of values for initializing:
    line = f_in.readline()
    parts = line.rstrip().split(",")
    current_icustay = parts[0]
    current_time = parts[2]
    header = parts[0:4]

    tmp_values = parts[4:len(parts)] # get list of all medical values as each as string

    current_values = np.repeat(np.nan, len(variable_indices)) # initialize current values to NANs

    # Process each value of the first line:
    for i in range(len(current_values)): # convert only available numbers as integer to current_values array
        if tmp_values[i] != '':
            current_values[i] = float(tmp_values[i])
            var_counter[i] += 1 #for each non-NAN value count it for each variable seperately
            if var_sum[i] != var_sum[i]: #check if it is a nan
                var_sum[i] = current_values[i] #set it to the new value, as np.nans stay nan when adding numbers
            else:
                var_sum[i] += current_values[i] # sum all observations of each variable up (seperately) to later build timepoint-wise average

    # Open the output file
    with open(outpath, 'w') as data_file: 
        data_writer = csv.writer(data_file, delimiter=',')

        data_writer.writerow(firstrow2write) # Write header information to first line of outfile
        
        #process the remaining lines:
        for line in f_in:
            # Strip of the \n and split the line
            parts = line.rstrip().split(",")
            # Get new id
            new_icustay = parts[0]
            new_time = parts[2]
            new_header = parts[0:4]

            # Check if patient or point in time have changed! if yes, compute average of the medical variables and write to outfile
            if (new_icustay != current_icustay) or (new_time != current_time):
                # for each entry of var_sum divide by var_counter iff number available (non-NAN)
                averages = np.repeat(np.nan, len(var_sum))
    
                for i in range(len(var_sum)):
                    if var_sum[i] == var_sum[i]: # if var_sum is NOT a NaN, compute average
                        averages[i] = var_sum[i]/var_counter[i]
                # write this array of averages (and potentially NANs) to a line of output file
                row2write = np.append(header, averages)
                data_writer.writerow(row2write)
                #f_out.write("%s,%s,%s\n" % (header ...))

                # reinitialise icustay_id, time and header such that next timepoint (and or patient) can be processed.
                current_icustay = new_icustay
                current_time = new_time
                header = new_header
                # reinitialise count and sum of variables for computing new averages:
                var_counter = np.zeros(len(variable_indices)) #counting how many observation for certain timepoint available (for averaging over)
                var_sum = np.repeat(np.nan, len(variable_indices)) # summing over the variables for certain timepoint (numerator for average)

        	# Process the current (patient, time) tuple!	
            tmp_values = parts[4:len(parts)]

            new_values = np.repeat(np.nan, len(variable_indices)) # initialize current values to NANs

            # Process each value of the line:
            for i in range(len(new_values)): # convert only available numbers to integer to current_values array
                if tmp_values[i] != '':
                    new_values[i] = float(tmp_values[i])
                    var_counter[i] += 1 #for each non-NAN value count it for each variable seperately
                    if var_sum[i] != var_sum[i]: #check if it is a nan
                        var_sum[i] = new_values[i] #set it to the new value, as np.nans stay nan when adding numbers
                    else:
                        var_sum[i] += new_values[i] # sum all observations of each variable up (seperately) to later build timepoint-wise average
    # Close the files
    f_in.close()

    end = time.time()
    print('Collecting records RUNTIME {} seconds'.format(end - start)) # Print runtime of this process

    return None









