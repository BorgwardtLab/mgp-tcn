/* Load sofa delta values and apply inclusion criteria to create Sepsis cohort based on hourly evaluated SOFA score.
Then extract time series values based on this Sepsis-Definition.

---------------------------------------------------------------------------------------------------------------------
AUTHOR: Michael Moor, February 2018.
---------------------------------------------------------------------------------------------------------------------

- output csv with sofa-delta and sepsis_onset_time in sofa_delta.csv, load it to db:
- IMPORT data from csv (created in python)

*/


drop table if exists sofa_delta;
create table sofa_delta(
    icustay_id   int,
    delta_score int,
    sepsis_onset timestamp,
    respiration     int,
    coagulation     int, 
    liver           int,
    cardiovascular  int,
    cns             int,
    renal           int,
    cv_Mean_BP_u70  int,
    cv_vasopressor_usage int

);

-- load sofa_delta.csv to sofa_delta table:    
\copy sofa_delta FROM '../../output/sofa_delta.csv' DELIMITER ',' CSV HEADER NULL ''



---------------------------------------------------
-- INCLUSION CRITERIA FOR DEFINING CASE/CONTROLS
---------------------------------------------------

\i hourly-cohort.sql 
-- write selected case and control ids in cases_hourly_ex1c, and controls_hourly
-- compute case cohort definitions by sequentially leaving out exclusion criteria for study count flow chart:
--\i check_exclusions/check_exclusions.sql -- cases_hourly_ex1b, _ex2 ... 

-- WRITE CASES_HOURLY TO CSV FOR CONTROL-MATCHING:
\copy ( SELECT * FROM cases_hourly_ex1c cv) To '../../output/q13_cases_hourly_ex1c.csv' with CSV HEADER;
\copy ( SELECT * FROM controls_hourly cv) To '../../output/q13_controls_hourly.csv' with CSV HEADER;



-- Call match-control.py script after this


