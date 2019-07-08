/*
Extract 55 hours of vital time series of sepsis case icustays (48 hours before onset and 7 hours after onset).
---------------------------------------------------------------------------------------------------------------------
- MODIFIED VERSION
- SOURCE: https://github.com/MIT-LCP/mimic-code/blob/7ff270c7079a42621f6e011de6ce4ddc0f7fd45c/concepts/firstday/vitals-first-day.sql
- AUTHOR (of this version): Michael Moor, October 2018
- HINT: to add/remove vitals uncomment/comment both the 'case' statements (e.g. l.35) the corresponding itemids below (e.g. l.112,113)
- CAVE: For our purposes we did not use CareVue data. Therefore their IDs were removed from the 'case' statement in l.66!
    However, you find them commented out in the 'where' clause at the end of the script.
---------------------------------------------------------------------------------------------------------------------
*/

--extract-time-series.sql
-- extract time series, TEMPLATE/inspiration: vitals-first-day.sql
-- info: to choose only 1 vital: comment out both the case statement (l.14) of the other variables and the corresponding itemids below (l.71-127)


DROP MATERIALIZED VIEW IF EXISTS case_55h_hourly_vitals_ex1c CASCADE;
create materialized view case_55h_hourly_vitals_ex1c as
SELECT  pvt.icustay_id, pvt.subject_id -- removed , pvt.hadm_id,
,   pvt.chart_time 
, case 
    when pvt.chart_time < pvt.sepsis_onset then 0 
    when pvt.chart_time between pvt.sepsis_onset and (pvt.sepsis_onset+interval '5' hour ) then 1
    else 2 end as sepsis_target
--, case
--  when pvt.sepsis_onset > (pvt.intime + interval '150' hour) then 1
--  else 0 end as late_onset_after_150h


-- Easier names

, case when VitalID = 2 then valuenum else null end as SysBP
, case when VitalID = 3 then valuenum else null end as DiaBP
, case when VitalID = 4 then valuenum else null end as MeanBP
, case when VitalID = 5 then valuenum else null end as RespRate
, case when VitalID = 1 then valuenum else null end as HeartRate
, case when VitalID = 7 then valuenum else null end as SpO2_pulsoxy
--, case when VitalID = 8 then valuenum else null end as Glucose
, case when VitalID = 6 then valuenum else null end as TempC
, case when VitalID = 10 then valuenum else null end as CardiacOutput
, case when VitalID = 11 then valuenum else null end as SV
, case when VitalID = 12 then valuenum else null end as SVI
, case when VitalID = 13 then valuenum else null end as SVV
, case when VitalID = 14 then valuenum else null end as TFC
, case when VitalID = 15 then valuenum else null end as TPR
, case when VitalID = 16 then valuenum else null end as TVset
, case when VitalID = 17 then valuenum else null end as TVobserved
, case when VitalID = 18 then valuenum else null end as TVspontaneous
, case when VitalID = 19 then valuenum else null end as Flowrate
, case when VitalID = 20 then valuenum else null end as PeakInspPressure
, case when VitalID = 21 then valuenum else null end as TotalPEEPLevel
, case when VitalID = 22 then valuenum else null end as VitalCapacity
, case when VitalID = 23 then valuenum else null end as O2Flow
, case when VitalID = 24 then valuenum else null end as FiO2
, case when VitalID = 25 then valuenum else null end as CRP

FROM  (
  select ch.icustay_id, ie.subject_id -- removed: ie.subject_id, ie.hadm_id, 
  , case
    when itemid in (220045) and valuenum > 0 and valuenum < 300 then 1 -- HeartRate
    when itemid in (220179,220050,225309) and valuenum > 0 and valuenum < 400 then 2 -- SysBP
    when itemid in (8368,8440,8441,8555,220180,220051,225310) and valuenum > 0 and valuenum < 300 then 3 -- DiasBP
    when itemid in (220052,220181,225312) and valuenum > 0 and valuenum < 300 then 4 -- MeanBP
    when itemid in (220210,224690) and valuenum > 0 and valuenum < 70 then 5 -- RespRate
    when itemid in (223761,678) and valuenum > 70 and valuenum < 120  then 6 -- TempF, converted to degC in valuenum call
    when itemid in (223762,676) and valuenum > 10 and valuenum < 50  then 6 -- TempC
    when itemid in (646,220277) and valuenum > 0 and valuenum <= 100 then 7 -- SpO2
    when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum > 0 then 8 -- Glucose
    --when itemid in (227428) and valuenum >= 0 then 9 -- SOFA score (computed in-ICU!)
    when itemid in (228369, 224842, 220088, 227543) and valuenum > 0 then 10 -- Cardiac Output (l/min), ids: NICOM, hemodynamics, hemodynamics Thermodilution, CO arterial (hemodynamics)
    when itemid in (228374, 227547) and valuenum > 0 then 11 -- SV: Stroke Volume (ml/beat) (id: NICOM, SV Arterial (Hemodynamics))
    when itemid in (228375) and valuenum >= 0 then 12 -- SVI-NICOM: Strove Volume Index (%) (id: NICOM)
    when itemid in (228376, 227546) then 13 -- SVV : Strove Volume Variation (no unit) (id: NICOM, SVV arterial (hemodynamics))
    when itemid in (228380) then 14 -- Thoracic Fluid Content (TFC)  (no unit) (NICOM)
    when itemid in (228381) then 15 -- Total Peripheral Resistance (TPR) (dynes*sec/cm5) (NICOM)
    when itemid in (224684) and valuenum > 0 then 16 -- Tidal Volume (set) (ml)
    when itemid in (224685) and valuenum > 0 then 17 -- Tidal Volume (observed) (ml)
    when itemid in (224686) and valuenum > 0 then 18 -- Tidal Volume (spontaneous) (ml)
    when itemid in (224691) and valuenum >= 0 then 19 -- Flow rate (L/min) (respiratory)
    when itemid in (224695) and valuenum >= 0 then 20 -- Peak Insp. Pressure metavision  chartevents Respiratory (cmH2O)
    when itemid in (224700) and valuenum >= 0 then 21 -- Total PEEP Level   metavision  chartevents Respiratory (cmH2O)
    when itemid in (220218) and valuenum > 0 then 22 --  Vital Capacity  VC  metavision  chartevents Respiratory (Liters)
    when itemid in (223834, 227582) and valuenum >= 0 then 23 -- O2 Flow O2 Flow metavision  chartevents Respiratory L/min (ids: respiratory, BIBAP)
    when itemid in (223835) and valuenum >= 0 then 24 -- Inspired O2 Fraction  FiO2  metavision  chartevents Respiratory (No unit)
    when itemid in (227444) and valuenum >= 0 then 25 -- CRP (no values in labevents! thus we use chartevents CRP)

    else null end as VitalID
      -- convert F to C
  , case when itemid in (223761,678) then (valuenum-32)/1.8 else valuenum end as valuenum
  , ce.charttime as chart_time
  , ch.sepsis_onset
  , s3c.intime

  from cases_hourly_ex1c ch -- was icustays ie (changed it below as well)
  left join icustays ie
    on ch.icustay_id = ie.icustay_id
  left join sepsis3_cohort s3c
    on ch.icustay_id = s3c.icustay_id
  left join chartevents ce
    on ch.icustay_id = ce.icustay_id -- removed: ie.subject_id = ce.subject_id and ie.hadm_id = ce.hadm_id and 
  and ce.charttime between (ch.sepsis_onset-interval '48' hour ) and (ch.sepsis_onset+interval '7' hour ) 

  -- exclude rows marked as error
  where ce.error=0 and
   ce.itemid in -- and sepsis_case = 1
  (
  ---- HEART RATE
  --211, --"Heart Rate"
  220045, --"Heart Rate"

  -- Systolic/diastolic

--  51, --  Arterial BP [Systolic]
--  442, -- Manual BP [Systolic]
--  455, -- NBP [Systolic]
--  6701, --    Arterial BP #2 [Systolic]
  220179, --    Non Invasive Blood Pressure systolic
  220050, --    Arterial Blood Pressure systolic
  225309, --    ART BP systolic

--  8368, --    Arterial BP [Diastolic]
--  8440, --    Manual BP [Diastolic]
--  8441, --    NBP [Diastolic]
--  8555, --    Arterial BP #2 [Diastolic]
  220180, --    Non Invasive Blood Pressure diastolic
  220051, --    Arterial Blood Pressure diastolic
  225310, --    ART BP diastolic


--  -- MEAN ARTERIAL PRESSURE
--  456, --"NBP Mean"
--  52, --"Arterial BP Mean"
--  6702, --    Arterial BP Mean #2
--  443, -- Manual BP Mean(calc)
  220052, --"Arterial Blood Pressure mean"
  220181, --"Non Invasive Blood Pressure mean"
  225312, --"ART BP mean"
--  224322, -- I-ABP mean  

  -- RESPIRATORY RATE
--  618,--  Respiratory Rate
--  615,--  Resp Rate (Total)
  220210,-- Respiratory Rate
  224690, --, --    Respiratory Rate (Total)


  -- SPO2, peripheral
  220277,

  -- GLUCOSE, both lab and fingerstick
--  807,--  Fingerstick Glucose
--  811,--  Glucose (70-105)
--  1529,-- Glucose
--  3745,-- BloodGlucose
--  3744,-- Blood Glucose
  225664,-- Glucose finger stick
  220621,-- Glucose (serum)
  226537,-- Glucose (whole blood)

--  -- TEMPERATURE
  223762, -- "Temperature Celsius"
--  676,    -- "Temperature C"
  223761, -- "Temperature Fahrenheit"
--  678 --  "Temperature F"

-- --SOFA SCORE (in icu)
-- 227428

-- Cardiac Output
  228369, -- NICOM,   
  224842, -- hemodynamics,
  220088, -- hemodynamics Thermodilution,
  227543, -- CO arterial (hemodynamics)

-- Stroke Volume
  228374, -- SV NICOM
  227547, -- SV Arterial (Hemodynamics)

-- Stroke Volume Index
  228375, -- SVI-NICOM: Strove Volume Index (%) NICOM

-- Stroke Volume Variation
  228376, -- SVV: NICOM
  227546, -- SVV: arterial (hemodynamics))

-- Thoracic Fluid Content 
  228380, -- TFC (no unit) (NICOM)

-- Total Peripheral Resistance
  228381, -- TPR (dynes*sec/cm5) (NICOM)

-- Tidal Volume set 
  224684, -- (ml)

-- Tidal Volume (observed)
  224685, --  (ml)

-- Tidal Volume (spontaneous)
  224686,  -- (ml)
  
-- Flow rate (respiratory)
  224691, -- (L/min) (respiratory)

-- Peak Insp. Pressure 
  224695, -- metavision  chartevents Respiratory (cmH2O)
  
-- Total PEEP Level 
  224700, -- Total PEEP Level   metavision  chartevents Respiratory (cmH2O)

-- Vital Capacity  VC
  220218, --  Vital Capacity  VC  metavision  chartevents Respiratory (Liters)

-- O2 Flow 
  223834, -- O2 Flow respiratory
  227582, -- O2 Flow BIBAP 
-- Inspired O2 Fraction  (FiO2)
  223835, -- Inspired O2 Fraction  FiO2  metavision  chartevents Respiratory (No unit)

  227444 -- C Reactive Protein (CRP) mg/L metavision  chartevents Labs (NO labevents! therefore use chartevents..)

  )
  
) pvt
--group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id
order by pvt.icustay_id, pvt.subject_id, pvt.chart_time; -- removed pvt.hadm_id, 




