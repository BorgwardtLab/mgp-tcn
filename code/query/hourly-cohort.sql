
/*
Create Case and Control Cohort based on Inclusion Criteria and Sepsis-3 Definition
---------------------------------------------------------------------------------------------------------------------
- AUTHOR Michael Moor, February 2018, last update: October 2018
---------------------------------------------------------------------------------------------------------------------
*/


-- create new hourly-case-cohort:
drop table if exists cases_hourly_ex1c CASCADE;
create table cases_hourly_ex1c(
    icustay_id   int,
    intime timestamp,
    outtime timestamp,
    length_of_stay double precision,
    delta_score int,
    sepsis_onset timestamp,
    sepsis_onset_day double precision,
    sepsis_onset_hour double precision
);


-- define cases and controls: join sofa-delta table with exclusion criteria!
insert into cases_hourly_ex1c
select sd.icustay_id
    , s3c.intime
    , s3c.outtime
    , extract(EPOCH from s3c.outtime - s3c.intime)
          / 60.0 / 60.0 as length_of_stay
    , sd.delta_score
    , sd.sepsis_onset
    , extract(EPOCH from sd.sepsis_onset - s3c.intime)
          / 60.0 / 60.0 / 24.0 as sepsis_onset_day
    , extract(EPOCH from sd.sepsis_onset - s3c.intime)
          / 60.0 / 60.0 as sepsis_onset_hour
        from sofa_delta sd
        inner join sepsis3_cohort s3c
            on sd.icustay_id = s3c.icustay_id
          inner join admissions adm
            on s3c.hadm_id = adm.hadm_id 
        where not (
                s3c.age <= 14 -- CHANGED FROM ORIGINAL! <=16
                    or adm.HAS_CHARTEVENTS_DATA = 0
                    or s3c.intime is null
                    or s3c.outtime is null
                    or s3c.dbsource != 'metavision'
                or s3c.suspected_of_infection_poe = 0
                 or sd.sepsis_onset > s3c.outtime
                 or sd.sepsis_onset < s3c.intime
                ); 


--new control cohort (without corrected icd criteria!)
drop table if exists controls_hourly CASCADE;
create table controls_hourly(
    icustay_id   int,
    hadm_id      int,
    intime timestamp,
    outtime timestamp,
    length_of_stay double precision,
    delta_score int,
    sepsis_onset timestamp
);

insert into controls_hourly

select s3c.icustay_id
    , s3c.hadm_id
    , s3c.intime
    , s3c.outtime
    , extract(EPOCH from s3c.outtime - s3c.intime)
          / 60.0 / 60.0 as length_of_stay
    , sd.delta_score
    , sd.sepsis_onset
        from sepsis3_cohort s3c
          left join sofa_delta sd 
            on s3c.icustay_id = sd.icustay_id
          inner join admissions adm
            on s3c.hadm_id = adm.hadm_id 
          -- NEW: to remove icd9 sepsis from controls! 

          where 
              s3c.hadm_id not in (
                select distinct(dg.hadm_id) 
                  from diagnoses_icd dg 
                    where dg.icd9_code in ('78552','99591','99592'))
              and not (
                    s3c.age <= 14 -- CHANGED FROM ORIGINAL! <=16
                    or adm.HAS_CHARTEVENTS_DATA = 0
                    or s3c.intime is null
                    or s3c.outtime is null
                    or s3c.dbsource != 'metavision'
                    or sd.sepsis_onset is not null
                    --or dg.icd9_code in ('78552','99591','99592') -- (Sept. Shock, Sepsis, Severe Sepsis)
              --or s3c.suspected_of_infection_poe = 0
              --or s3c.inclusion_suspicion_after_4_hours = 0 
              --or s3c.suspected_infection_time_poe > s3c.outtime  -- those are the exclusion criteria used in case-control.sql!
                );





