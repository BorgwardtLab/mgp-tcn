/* main_query.sql 

---------------------------------------------------------------------------------------------------------------------
AUTHOR: Michael Moor, February 2018.
---------------------------------------------------------------------------------------------------------------------


> Big parts of this code is based on this source:
==> https://github.com/alistairewj/sepsis3-mimic
Johnson, Alistair EW, David J. Stone, Leo A. Celi, and Tom J. Pollard. 
"The MIMIC Code Repository: enabling reproducibility in critical care research." 
Journal of the American Medical Informatics Association (2017): ocx084. 

> This query creates all required tables and performs a sepsis-3 definition by employing an hourly SOFA
evaluation within [-48,+24] hours of suspicion of infection time. Therefore, subsequently the SOFA computation is looped over 72 hours.
Based on this a Case/Control Definition is implemented. For both vital time series are then extracted and stored in csv format.

*/

---------------------------------------------------
-- CREATE COHORT TABLE (identical to alistairewi)
---------------------------------------------------

-- identify if there is suspicion of infection. If yes determine time of inf-susp (for all available icustays!) 	
\i abx-poe-list.sql
\i abx-micro-prescription.sql
\i suspicion-of-infection.sql

-- generate cohort (featuring susp-inf-time, age, sex, ..)
\i cohort.sql


---------------------------------------------------
-- FURTHER PREREQUISITES FOR QUERY 
---------------------------------------------------

-- create tables to fill with clinical data in subsequent loop over the 72hours:
\i create_tables.sql

-- define all functions that extract time-sensitive variables required for later hourly sofa score computation
	-- (alistairewi scripts but with different time windows (-24h to now) to selected chartevents)
\i hourly-urine-output-infect-time.sql
\i hourly-vitals-infect-time.sql
\i hourly-gcs-infect-time.sql
\i hourly-labs-infect-time.sql 
\i hourly-blood-gas-arterial-infect-time.sql

-- extract information about mechanical ventilation duration for sofa-hourly.sql (original version of alistairewi)
\i ventilation-durations.sql


-- define function that computes the SOFA of the last 24h for a given hour 
	-- (integer relative to 48h before suspicion of infection (refered to as si_starttime ))   
	-- and write result to the created tables (of l.34)
	-- this script is a modified version of alistairewis sofa-si.sql
\i sofa-hourly.sql

-- define loop function that feeds the sofa computing scripts the relative hour from 0 to 72 after si_starttime
\i loop.sql


---------------------------------------------------
-- PERFORM ACTUAL SOFA COMPUTATION 
---------------------------------------------------

--Looping over the 72h with DO by default as a pl/pgsql block:
DO $$ BEGIN
    PERFORM sofa_loop();
END $$;


---------------------------------------------------
-- COMPUTE SOFA CHANGE (off-line in python)
---------------------------------------------------

-- the delta-sofa extraction will be performed in python for runtime and dynamic programming reasons
-- for that EXPORT data to csv
\copy ( SELECT * FROM sofa_table order by icustay_id, time_window) To '../../output/sofa_table.csv' with CSV HEADER;
\copy ( SELECT icustay_id, si_starttime FROM suspinfect_poe) To '../../output/si_starttime.csv' with CSV HEADER;

