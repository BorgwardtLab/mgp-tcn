/*
Determine urine output of the last 24 hours on a hourly base
---------------------------------------------------------------------------------------------------------------------
- MODIFIED VERSION
- SOURCE: https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/urine-output-infect-time.sql
- AUTHOR (of this version): Michael Moor, February 2018
---------------------------------------------------------------------------------------------------------------------
*/

DROP FUNCTION IF EXISTS get_uo(int);
CREATE FUNCTION get_uo(int) returns void
    AS 
    $$
      insert into uo_table 
      select
  -- patient identifiers
  si.icustay_id

  -- volumes associated with urine output ITEMIDs
  , case
      when max(charttime) = min(charttime)
        then null
      when max(charttime) is not null
        then sum(VALUE)
          / (extract(EPOCH from (max(charttime) - min(charttime)))/60.0/60.0/24.0)
    else null end
    as UrineOutput -- daily urine output
  , $1 as time_window
     
from suspinfect_poe si  --CHANGED FROM ORIGINAL: added _poe!
-- Join to the outputevents table to get urine output
left join outputevents oe
-- join on all patient identifiers
on si.icustay_id = oe.icustay_id
-- and ensure the data occurs during the ICU stay
and oe.charttime
--    between (si.si_starttime + $1 * interval '1 hour') - interval '72' hour  -- try 3d sliding window!
--        and si.si_starttime + $1 * interval '1 hour'
--        between si.si_starttime - interval '24' hour -- try out new 1d window but hardcoded!
--        and si.si_starttime
--      between si.si_starttime -- try out old 3d window!
--        and si.si_endtime
  between (si.si_starttime + $1 * interval '1 hour') - interval '24' hour  
        and si.si_starttime + $1 * interval '1 hour'    
where itemid in
(
-- these are the most frequently occurring urine output observations in CareVue
40055, -- "Urine Out Foley"
43175, -- "Urine ."
40069, -- "Urine Out Void"
40094, -- "Urine Out Condom Cath"
40715, -- "Urine Out Suprapubic"
40473, -- "Urine Out IleoConduit"
40085, -- "Urine Out Incontinent"
40057, -- "Urine Out Rt Nephrostomy"
40056, -- "Urine Out Lt Nephrostomy"
40405, -- "Urine Out Other"
40428, -- "Urine Out Straight Cath"
40086,--	Urine Out Incontinent
40096, -- "Urine Out Ureteral Stent #1"
40651, -- "Urine Out Ureteral Stent #2"

-- these are the most frequently occurring urine output observations in Metavision
226559, -- "Foley"
226560, -- "Void"
227510, -- "TF Residual"
226561, -- "Condom Cath"
226584, -- "Ileoconduit"
226563, -- "Suprapubic"
226564, -- "R Nephrostomy"
226565, -- "L Nephrostomy"
226567, --	Straight Cath
226557, -- "R Ureteral Stent"
226558  -- "L Ureteral Stent"
)
group by si.icustay_id
order by si.icustay_id;

    $$
    LANGUAGE SQL;
