/* Second part of main_write query (after case-control matching in python)

---------------------------------------------------------------------------------------------------------------------
AUTHOR: Michael Moor, October 2018.
---------------------------------------------------------------------------------------------------------------------


-- Called match-control.py script before this one. Now load matched_controls.csv:

*/


drop table if exists matched_controls_hourly CASCADE;
create table matched_controls_hourly(
    icustay_id          int,
    hadm_id             int,
    intime              timestamp,
    outtime            timestamp,
    length_of_stay      numeric,
    control_onset_hour  numeric,
    control_onset_time  timestamp,
    matched_case_icustay_id int
);
-- new csv file columns: icustay_id,hadm_id,intime,outtime,length_of_stay,control_onset_hour,control_onset_time,matched_case_icustay_id

-- load q13_matched_controls.csv into matched_controls_hourly table in psql (required for control matching, control onset)  
\copy matched_controls_hourly FROM '../../output/q13_matched_controls.csv' DELIMITER ',' CSV HEADER NULL ''

---------------------------------------------------
-- EXTRACT VITAL TIME SERIES FOR CASES AND CONTROLS FROM CHARTEVENTS (few lab exceptions)
---------------------------------------------------
-- here, alistairewis scripts were used and modified to extract the desired variables and chart windows.  
-- for adding / removing clinical variables adjust here
\i extract-55h-of-hourly-case-vital-series_ex1c.sql
\i extract-55h-of-hourly-control-vital-series_ex1c.sql

---------------------------------------------------
-- EXTRACT LAB TIME SERIES FOR CASES AND CONTROLS
---------------------------------------------------
\i extract-55h-of-hourly-case-lab-series_ex1c.sql
\i extract-55h-of-hourly-control-lab-series_ex1c.sql

---------------------------------------------------
-- GET STATIC VARIABLES FOR CASES AND CONTROLS and write to csv
---------------------------------------------------
\i static-query.sql

---------------------------------------------------
-- WRITE QUERIED DATA TO CSV
---------------------------------------------------

-- WRITE STATIC COVARIATES (age, gender, ..) INTO STATIC CSV
\copy (select * from icustay_static) To '../../output/static_variables.csv' with CSV HEADER;
\copy (select * from icustay_static st inner join cases_hourly_ex1c ch on st.icustay_id=ch.icustay_id) To '../../output/static_variables_cases.csv' with CSV HEADER;
\copy (select * from icustay_static st inner join matched_controls_hourly ch on st.icustay_id=ch.icustay_id) To '../../output/static_variables_controls.csv' with CSV HEADER;

-- Write vitals to csv:
\copy (select * from case_55h_hourly_vitals_ex1c cv order by cv.icustay_id, cv.chart_time) To '../../output/case_55h_hourly_vitals_ex1c.csv' with CSV HEADER;
\copy (select * from control_55h_hourly_vitals_ex1c cv order by cv.icustay_id, cv.chart_time) To '../../output/control_55h_hourly_vitals_ex1c.csv' with CSV HEADER;
-- Write labs to csv:
\copy (select * from case_55h_hourly_labs_ex1c cl order by cl.icustay_id, cl.chart_time) To '../../output/case_55h_hourly_labs_ex1c.csv' with CSV HEADER;
\copy (select * from control_55h_hourly_labs_ex1c cl order by cl.icustay_id, cl.chart_time) To '../../output/control_55h_hourly_labs_ex1c.csv' with CSV HEADER;


