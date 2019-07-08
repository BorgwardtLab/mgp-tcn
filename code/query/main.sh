#!/bin/sh

#Main Script to call all psql queries. 

#################################################################
#DEFINE DATABASE USERNAME: 
database=$1
username=$2   
#Define paths for Python script
prj_dir=../.. 

# Check if prj_dir exists, else exit with error 
if [ ! -d "$prj_dir" ]; then
    echo 'Error: the entered directory was not found' > logfile.log 
    exit 1   
fi

if [ ! -d "$prj_dir/code" ]; then
    echo 'create code directory' >> logfile.log    
    mkdir $prj_dir/code
fi
if [ ! -d "$prj_dir/code/query" ]; then
    echo 'create query directory' >> logfile.log
    mkdir $prj_dir/code/query
fi

script_dir=$prj_dir/code/query

if [ ! -d "$prj_dir/output" ]; then
    echo 'create output directory' >> logfile.log    
    mkdir $prj_dir/output
fi

out_dir=$prj_dir/output

echo 'Starting QUERY ...' 

#Sepsis-3 Query:

# Run main-query.sql (first part of main.sql up to python)

cmd="dbname=${database} user=${username} options=--search_path=mimiciii"
psql "$cmd" -f main_query.sql

# Run the Python script as this step is easier in python:
python3 $script_dir/compute_sepsis_onset_from_exported_sql_table.py \
    --file_timepoint $out_dir/sofa_table.csv \
    --file_ref_time $out_dir/si_starttime.csv \
    --sofa_threshold 2 \
    --file_output $out_dir/sofa_delta.csv

# Run main-write.sql (second part of main.sql after python)
psql "$cmd" -f main_write.sql


# Intermediate dynamic Python processing step to get case-control matching:
python3 match-controls.py \
    --casefile $out_dir/q13_cases_hourly_ex1c.csv \
    --controlfile $out_dir/q13_controls_hourly.csv \
    --outfile $out_dir/q13_matched_controls.csv

# Given case controls are assigned, extract relevant time windows
# Run second part of main-write (now main-write2.sql)
psql "$cmd" -f main_write2.sql



