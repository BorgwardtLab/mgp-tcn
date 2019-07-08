/*
Assemble required variables and compute SOFA score
---------------------------------------------------------------------------------------------------------------------
- MODIFIED VERSION
- SOURCE: https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/sofa-si.sql
- AUTHOR (of this version): Michael Moor, February 2018
---------------------------------------------------------------------------------------------------------------------
*/

-- This script extracts the hourly SOFA score of the previous 24 hours as a function of 
-- the hour count within the 72h suspected-infection-window!

-- This query extracts the sequential organ failure assessment (formally: sepsis-related organ failure assessment).
-- This score is a measure of organ failure for patients in the ICU.

-- Reference for SOFA:
--    Jean-Louis Vincent, Rui Moreno, Jukka Takala, Sheila Willatts, Arnaldo De Mendonça,
--    Hajo Bruining, C. K. Reinhart, Peter M Suter, and L. G. Thijs.
--    "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure."
--    Intensive care medicine 22, no. 7 (1996): 707-710.

-- Variables used in SOFA:
--  GCS, MAP, FiO2, Ventilation status (sourced from CHARTEVENTS)
--  Creatinine, Bilirubin, FiO2, PaO2, Platelets (sourced from LABEVENTS)
--  Dobutamine, Epinephrine, Norepinephrine (sourced from INPUTEVENTS_MV and INPUTEVENTS_CV)
--  Urine output (sourced from OUTPUTEVENTS)

-- This script also presumes ventdurations!

DROP FUNCTION IF EXISTS get_vaso_mv(int);
CREATE FUNCTION get_vaso_mv(int) returns void
    AS 
    $$
      insert into vaso_mv_table 
      select ie.icustay_id
    -- case statement determining whether the ITEMID is an instance of vasopressor usage
    , max(case when itemid = 221906 then rate end) as rate_norepinephrine
    , max(case when itemid = 221289 then rate end) as rate_epinephrine
    , max(case when itemid = 221662 then rate end) as rate_dopamine
    , max(case when itemid = 221653 then rate end) as rate_dobutamine
    , $1 as time_window -- insert also time_window!

  from suspinfect_poe s  --CHANGED FROM ORIGINAL: added _poe!
  inner join icustays ie
    on s.icustay_id = ie.icustay_id
  inner join inputevents_mv mv
    on ie.icustay_id = mv.icustay_id
    and mv.starttime
    between (s.si_starttime + $1 * interval '1 hour') - interval '24' hour  
        and s.si_starttime + $1 * interval '1 hour'

  where itemid in (221906,221289,221662,221653)
  -- 'Rewritten' orders are not delivered to the patient
  and statusdescription != 'Rewritten'
  group by ie.icustay_id  
    ; 
    $$
    LANGUAGE SQL;


DROP FUNCTION IF EXISTS get_pafi1(int); --requires bg_table of current time_window!
CREATE FUNCTION get_pafi1(int) returns void
    AS 
    $$
    insert into pafi1_table
        -- join blood gas to ventilation durations to determine if patient was vent
    select bg.icustay_id
    , PaO2FiO2
    , case when vd.icustay_id is not null then 1 else 0 end as IsVent
    , $1 as time_window -- insert also time_window!
    from bg_table bg -- used to be bloodgasarterial (now a time-sensitive table!)
    left join ventdurations vd -- not time-sensitive!
      on bg.icustay_id = vd.icustay_id
      and bg.charttime >= vd.starttime
      and bg.charttime <= vd.endtime
    where bg.time_window = $1  -- choose only current time_window!
    order by bg.icustay_id, bg.charttime
    ; 
    $$
    LANGUAGE SQL;


DROP FUNCTION IF EXISTS get_pafi2(int); --requires pafi1_table of current time_window!
CREATE FUNCTION get_pafi2(int) returns void
    AS 
    $$
    insert into pafi2_table
    -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
    -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
    -- in this case, the SOFA score is 3, *not* 4.
    select icustay_id
    , min(case when IsVent = 0 then PaO2FiO2 else null end) as PaO2FiO2_novent_min
    , min(case when IsVent = 1 then PaO2FiO2 else null end) as PaO2FiO2_vent_min
    , $1 as time_window-- insert also time_window!
    from pafi1_table pf1
      where pf1.time_window = $1
      group by icustay_id
    ; 
    $$
    LANGUAGE SQL;



DROP FUNCTION IF EXISTS score_compute(int); --requires ...
CREATE FUNCTION score_compute(int) returns void
    AS 
    $$
    with scorecomp as
    (

    -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
    -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
    -- in this case, the SOFA score is 3, *not* 4.
    
    select ie.icustay_id
  , v.MeanBP_Min
 
  , mv.rate_norepinephrine 
  , mv.rate_epinephrine 
  , mv.rate_dopamine 
  , mv.rate_dobutamine 

  , l.Creatinine_Max
  , l.Bilirubin_Max
  , l.Platelet_Min
 
  , pf.PaO2FiO2_novent_min
  , pf.PaO2FiO2_vent_min

  , uo.UrineOutput

  , gcs.MinGCS

from suspinfect_poe s  --CHANGED FROM ORIGINAL: added _poe!
inner join icustays ie  --not time-sensitive
  on s.icustay_id = ie.icustay_id
--left join vaso_cv cv -- we don't use carevue logging!
--  on ie.icustay_id = cv.icustay_id
left join vaso_mv_table mv  -- *_table are time-sensitive (check time window)!
  on ie.icustay_id = mv.icustay_id
left join pafi2_table pf  
 on ie.icustay_id = pf.icustay_id
left join vitals_table v 
  on ie.icustay_id = v.icustay_id
left join labs_table l 
  on ie.icustay_id = l.icustay_id
left join uo_table uo 
  on ie.icustay_id = uo.icustay_id
left join gcs_table gcs
  on ie.icustay_id = gcs.icustay_id

  where v.time_window = $1
   and mv.time_window = $1
   and pf.time_window = $1
   and l.time_window = $1
   and uo.time_window = $1
   and gcs.time_window = $1
                 
  )
  , scorecalc as
  (
       select icustay_id
  -- Respiration
  , case
      when PaO2FiO2_vent_min   < 100 then 4
      when PaO2FiO2_vent_min   < 200 then 3
      when PaO2FiO2_novent_min < 300 then 2
      when PaO2FiO2_novent_min < 400 then 1
      when coalesce(PaO2FiO2_vent_min, PaO2FiO2_novent_min) is null then null
      else 0
    end as respiration

  -- Coagulation
  , case
      when platelet_min < 20  then 4
      when platelet_min < 50  then 3
      when platelet_min < 100 then 2
      when platelet_min < 150 then 1
      when platelet_min is null then null
      else 0
    end as coagulation

  -- Liver
  , case
      -- Bilirubin checks in mg/dL
        when Bilirubin_Max >= 12.0 then 4
        when Bilirubin_Max >= 6.0  then 3
        when Bilirubin_Max >= 2.0  then 2
        when Bilirubin_Max >= 1.2  then 1
        when Bilirubin_Max is null then null
        else 0
      end as liver

  -- Cardiovascular
  , case
      when rate_dopamine > 15 or rate_epinephrine >  0.1 or rate_norepinephrine >  0.1 then 4
      when rate_dopamine >  5 or (rate_epinephrine <= 0.1 AND rate_epinephrine > 0 ) or (rate_norepinephrine <= 0.1 AND rate_norepinephrine > 0) then 3
      when rate_dopamine >  0 or rate_dobutamine > 0 then 2
      when MeanBP_Min < 70 then 1
      when coalesce(MeanBP_Min, rate_dopamine, rate_dobutamine, rate_epinephrine, rate_norepinephrine) is null then null
      else 0
    end as cardiovascular

  -- Neurological failure (GCS)
  , case
      when (MinGCS >= 13 and MinGCS <= 14) then 1
      when (MinGCS >= 10 and MinGCS <= 12) then 2
      when (MinGCS >=  6 and MinGCS <=  9) then 3
      when  MinGCS <   6 then 4
      when  MinGCS is null then null
  else 0 end
    as cns

  -- Renal failure - high creatinine or low urine output
  , case
    when (Creatinine_Max >= 5.0) then 4
    when  UrineOutput < 200 then 4
    when (Creatinine_Max >= 3.5 and Creatinine_Max < 5.0) then 3
    when  UrineOutput < 500 then 3
    when (Creatinine_Max >= 2.0 and Creatinine_Max < 3.5) then 2
    when (Creatinine_Max >= 1.2 and Creatinine_Max < 2.0) then 1
    when coalesce(UrineOutput, Creatinine_Max) is null then null
  else 0 end
    as renal

  -- Study Label Contamination through Features by analyzing sofa contributing features
  , case
    when MeanBP_Min < 70 then 1
  else 0 end
    as cv_Mean_BP_u70

  , case
    when rate_dopamine >  0 or rate_dobutamine > 0 or rate_epinephrine > 0 or rate_norepinephrine >  0 then 1
  else 0 end
    as cv_vasopressor_usage


      

    from scorecomp 
)
, score_final as 
(
select si.icustay_id
        -- Combine all the scores to get SOFA
        -- Impute 0 if the score is missing
        , coalesce(respiration,0)
        + coalesce(coagulation,0)
        + coalesce(liver,0)
        + coalesce(cardiovascular,0)
        + coalesce(cns,0)
        + coalesce(renal,0)
        as SOFA
        , respiration
        , coagulation
        , liver
        , cardiovascular
        , cns
        , renal
        , cv_Mean_BP_u70
        , cv_vasopressor_usage

from suspinfect_poe si  --CHANGED FROM ORIGINAL: added _poe!
left join scorecalc s
  on si.icustay_id = s.icustay_id
where si.suspected_infection_time is not null
order by si.icustay_id
)

insert into sofa_table
select  icustay_id
    , SOFA
    , $1 as time_window
    , respiration
    , coagulation
    , liver
    , cardiovascular
    , cns
    , renal
    , cv_Mean_BP_u70
    , cv_vasopressor_usage
    
    from score_final; 
    $$
    LANGUAGE SQL;





