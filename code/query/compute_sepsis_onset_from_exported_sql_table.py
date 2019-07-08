#-----------------------------------------------------------------------------
# Determine the overlap among .bim files
#
# February 2018 M. Moor and D. Roqueiro
#-----------------------------------------------------------------------------
import sys
import os
import argparse
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
# Set up the parsing of command-line arguments
parser = argparse.ArgumentParser(description="Post-process the exported output files from the SQL query <ENTER_NAME>")
parser.add_argument("--file_timepoint", required=True, 
                    help="Full path to the file containing the sofa score at each timepoint")
parser.add_argument("--file_ref_time", required=True, 
                    help="Full path to the file containing the start time (date+time) of the timepoints in file_timepoint")
parser.add_argument("--sofa_threshold", required=True, type=int,
                    help="Threshold for the sofa score (score - threshold is considered an event)")
parser.add_argument("--file_output", required=True,
                    help="Full path to the output file")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Step 1. Create a dictionary with the star time per icustay
#         The id is the icustay
# Format of the file is (comma-separated)
#   icustay_id,si_starttime
#   200001,2181-11-16 11:10:00
try:
    f_in = open(args.file_ref_time, 'r')
except IOError:
    print("Cannot read input file %s" % args.file_ref_time)
    sys.exit(1)

# Initialize a dictionary
icustay_dict = {}

# Read the file, line by line
# Exclude the header
f_in.readline()
for line in f_in:
    # Split the line
    parts = line.split(",")
    # Get the id (first column) and use as key
    icustay_id = parts[0]
    icustay_dict[icustay_id] = parts[1].rstrip()
# Close the file
f_in.close()

# -----------------------------------------------------------------------------
# Step 2. Create the output file
try:
    f_out = open(args.file_output, 'w')
except IOError:
    print("Cannot create the output file %s" % args.file_output)
    sys.exit(1)

# -----------------------------------------------------------------------------
# Step 3. Process all the time points with a control-break per patient
#
# Format of the file is (comma-separated):
#   icustay_id,sofa,time_window
#   200001,0,0
try:
    f_in = open(args.file_timepoint, 'r')
except IOError:
    print("Cannot read input file %s" % args.file_timepoint)
    sys.exit(1)

# Iterate through all the timepoints. Break per icustay
# Exclude the header
f_in.readline()

# Read the first timepoint for the first icustay
line = f_in.readline()
# Split the line
parts = line.split(",")
current_icustay = parts[0]
min_sofa = int(parts[1])
min_timepoint = int(parts[2])
b_delta_fulfilled = False
line2write = [] #list of values to write to outfile

# Process the remaining lines
for line in f_in:
    # Split the line
    parts = line.split(",")
    # Get the fields
    icustay_id = parts[0]
    sofa_score = int(parts[1])
    timepoint  = int(parts[2])

    # Check if we changed to the next icustay
    if current_icustay != parts[0]:
        # We changed to the next icustay
        # Check if the condition from the previous icustay was fulfilled
        if b_delta_fulfilled:
            # Compute the time of sepsis onset
            start_time = datetime.strptime(icustay_dict[current_icustay], "%Y-%m-%d %H:%M:%S")
            onset_time = start_time + timedelta(hours = t_fulfilled)

            line2write = []
            line2write.append(current_icustay)
            line2write.append(delta_sofa_score)
            line2write.append(onset_time.strftime("%Y-%m-%d %H:%M:%S"))  
            line2write.append(curr_respiration) #curr_respiration = parts[3]
            line2write.append(curr_coagulation) #curr_coagulation = parts[4]
            line2write.append(curr_liver) #curr_liver = parts[5]
            line2write.append(curr_cardiovascular) #curr_cardiovascular = parts[6]
            line2write.append(curr_cns) #curr_cns = parts[7] 
            line2write.append(curr_renal) #curr_renal = parts[8]
            line2write.append(curr_cv_Mean_BP_u70) #curr_cv_Mean_BP_u70 = parts[9]
            line2write.append(curr_cv_vasopressor_usage) #curr_cv_vasopressor_usage = parts[10]
            writerow = ','.join([str(i) for i in line2write])    

            # Save to the output file
            f_out.write(writerow+'\r'.replace('\r', '')) #\n
        # Reset all variables
        b_delta_fulfilled = False
        current_icustay = parts[0]
        min_sofa = int(parts[1])
        min_timepoint = int(parts[2])
        line2write = []

    else:
        # Same icustay as before
        # First, check if the condition has been fulfilled. In that case, skip        
        if b_delta_fulfilled:
            continue;

        # The condition was not fulfilled yet.

        # Compute, for the icustay, the minimum sofa score so far
        if sofa_score < min_sofa:
            # Update the minimum
            min_sofa = sofa_score
            min_timepoint = timepoint
        else:
            # Determine if the difference is larger than the threshold (command-line argument)
            delta_sofa_score = sofa_score - min_sofa
            if delta_sofa_score >= args.sofa_threshold:
                # Detected.
                # Set boolean flag to skip processing the rest of the timepoints for the icustay
                b_delta_fulfilled = True
                t_fulfilled = timepoint

                # Get sofa contributions (--> update these variables inside if/else blocks as we don't want later values after b_delta_fulfilled, where steps are skipped with 'continue')
            
                curr_respiration = parts[3]
                curr_coagulation = parts[4]
                curr_liver = parts[5]
                curr_cardiovascular = parts[6]
                curr_cns = parts[7] 
                curr_renal = parts[8]
                curr_cv_Mean_BP_u70 = parts[9]
                curr_cv_vasopressor_usage = parts[10]
        
# Must process the last icustay
# Check if the condition from the previous icustay was fulfilled
if b_delta_fulfilled:
    # Compute the time of sepsis onset
    start_time = datetime.strptime(icustay_dict[current_icustay], "%Y-%m-%d %H:%M:%S")
    onset_time = start_time + timedelta(hours = t_fulfilled)

    line2write = []
    line2write.append(current_icustay)
    line2write.append(delta_sofa_score)
    line2write.append(onset_time.strftime("%Y-%m-%d %H:%M:%S"))  
    line2write.append(curr_respiration) #curr_respiration = parts[3]
    line2write.append(curr_coagulation) #curr_coagulation = parts[4]
    line2write.append(curr_liver) #curr_liver = parts[5]
    line2write.append(curr_cardiovascular) #curr_cardiovascular = parts[6]
    line2write.append(curr_cns) #curr_cns = parts[7] 
    line2write.append(curr_renal) #curr_renal = parts[8]
    line2write.append(curr_cv_Mean_BP_u70) #curr_cv_Mean_BP_u70 = parts[9]
    line2write.append(curr_cv_vasopressor_usage) #curr_cv_vasopressor_usage = parts[10]
    writerow = ','.join([str(i) for i in line2write])

    # Save to the output file
    f_out.write(writerow + '\r'.replace('\r', '')) # \n
# Close the file
f_out.close()

