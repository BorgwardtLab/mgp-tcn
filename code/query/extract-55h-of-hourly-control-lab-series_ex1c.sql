/*
Extract 55 hours of LAB time series of CONTROL icustays (48 hours before 'control-onset' and 7 hours after control-onset).
---------------------------------------------------------------------------------------------------------------------
- MODIFIED VERSION
- SOURCE: https://github.com/MIT-LCP/mimic-code/blob/7ff270c7079a42621f6e011de6ce4ddc0f7fd45c/concepts/firstday/lab-first-day.sql
- AUTHOR (of this version): Michael Moor, October 2018
---------------------------------------------------------------------------------------------------------------------
*/


-- This query pivots lab values 

DROP MATERIALIZED VIEW IF EXISTS control_55h_hourly_labs_ex1c CASCADE;
CREATE materialized VIEW control_55h_hourly_labs_ex1c AS
SELECT
  pvt.icustay_id
  , pvt.subject_id
  , pvt.chart_time
  , case 
    when pvt.chart_time < pvt.control_onset_time then 0 
    when pvt.chart_time between pvt.control_onset_time and (pvt.control_onset_time+interval '5' hour ) then 1
    else 2 end as pseudo_target 

  , CASE WHEN label = 'ANION GAP' THEN valuenum ELSE null END as ANIONGAP
  , CASE WHEN label = 'ALBUMIN' THEN valuenum ELSE null END as ALBUMIN
  , CASE WHEN label = 'BANDS' THEN valuenum ELSE null END as BANDS
  , CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE null END as BICARBONATE
  , CASE WHEN label = 'BILIRUBIN' THEN valuenum ELSE null END as BILIRUBIN
  , CASE WHEN label = 'CREATININE' THEN valuenum ELSE null END as CREATININE
  , CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE null END as CHLORIDE
  , CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE null END as GLUCOSE
  , CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE null END as HEMATOCRIT
  , CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE null END as HEMOGLOBIN
  , CASE WHEN label = 'LACTATE' THEN valuenum ELSE null END as LACTATE
  , CASE WHEN label = 'PLATELET' THEN valuenum ELSE null END as PLATELET
  , CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE null END as POTASSIUM
  , CASE WHEN label = 'PTT' THEN valuenum ELSE null END as PTT
  , CASE WHEN label = 'INR' THEN valuenum ELSE null END as INR
  , CASE WHEN label = 'PT' THEN valuenum ELSE null END as PT -- Prothrombin time
  , CASE WHEN label = 'SODIUM' THEN valuenum ELSE null END as SODIUM
  , CASE WHEN label = 'BUN' THEN valuenum ELSE null end as BUN
  , CASE WHEN label = 'WBC' THEN valuenum ELSE null end as WBC
  --, CASE WHEN label = 'CRP' THEN valuenum ElSE null end as CRP   -- CRP: very few values in labevents (such that psql has problems displaying it correctly (of cases at least) with heterogeneous uom --> use chartevents as too hard to debug this way..!
  , CASE WHEN label = 'Ferritin' THEN valuenum ELSE null END as Ferritin
  , CASE WHEN label = 'Transferrin'  THEN valuenum ELSE null END as Transferrin
  , CASE WHEN label = 'CreatineKinase' THEN valuenum ELSE null END as CreatineKinase
  , CASE WHEN label = 'CK_MB' THEN valuenum ELSE null END as CK_MB
  , CASE WHEN label = 'D_Dimer' THEN valuenum ELSE null END as D_Dimer
  , CASE WHEN label = 'NTproBNP' THEN valuenum ELSE null END as NTproBNP
  , CASE WHEN label = 'SedimentationRate' THEN valuenum ELSE null END as SedimentationRate
  , CASE WHEN label = 'Fibrinogen' THEN valuenum ELSE null END as Fibrinogen
  , CASE WHEN label = 'LDH' THEN valuenum ELSE null END as LDH
  , CASE WHEN label = 'Magnesium' THEN valuenum ELSE null END as Magnesium
  , CASE WHEN label = 'Calcium_free' THEN valuenum ELSE null END as Calcium_free
  , CASE WHEN label = 'pO2_bloodgas' THEN valuenum ELSE null END as pO2_bloodgas
  , CASE WHEN label = 'pH_bloodgas' THEN valuenum ELSE null END as pH_bloodgas
  , CASE WHEN label = 'pCO2_bloodgas' THEN valuenum ELSE null END as pCO2_bloodgas
  , CASE WHEN label = 'SO2_bloodgas' THEN valuenum ELSE null END as SO2_bloodgas
  , CASE WHEN label = 'Troponin_T' THEN valuenum ELSE null END as Troponin_T
  , CASE WHEN label = 'Glucose_CSF' THEN valuenum ELSE null END as Glucose_CSF
  , CASE WHEN label = 'Total_Protein_Joint_Fluid' THEN valuenum ELSE null END as Total_Protein_Joint_Fluid
  , CASE WHEN label = 'Total_Protein_Pleural' THEN valuenum ELSE null END as Total_Protein_Pleural
  , CASE WHEN label = 'Urine_Albumin_Creatinine_ratio' THEN valuenum ELSE null END as Urine_Albumin_Creatinine_ratio
  , CASE WHEN label = 'WBC_Ascites' THEN valuenum ELSE null END as WBC_Ascites


FROM
( -- begin query that extracts the data
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
  , le.charttime as chart_time
  , ch.control_onset_time
  , le.valueuom

  -- here we assign labels to ITEMIDs
  -- this also fuses together multiple ITEMIDs containing the same data
  , CASE
        WHEN itemid = 50868 THEN 'ANIONGAP'
        WHEN itemid = 50862 THEN 'ALBUMIN'
        WHEN itemid = 51144 THEN 'BANDS'
        WHEN itemid = 50882 THEN 'BICARBONATE'
        WHEN itemid = 50885 THEN 'BILIRUBIN'
        WHEN itemid = 50912 THEN 'CREATININE'
        WHEN itemid = 50806 THEN 'CHLORIDE'
        WHEN itemid = 50902 THEN 'CHLORIDE'
        WHEN itemid = 50809 THEN 'GLUCOSE'
        WHEN itemid = 50931 THEN 'GLUCOSE'
        WHEN itemid = 50810 THEN 'HEMATOCRIT'
        WHEN itemid = 51221 THEN 'HEMATOCRIT'
        WHEN itemid = 50811 THEN 'HEMOGLOBIN'
        WHEN itemid = 51222 THEN 'HEMOGLOBIN'
        WHEN itemid = 50813 THEN 'LACTATE'
        WHEN itemid = 51265 THEN 'PLATELET'
        WHEN itemid = 50822 THEN 'POTASSIUM'
        WHEN itemid = 50971 THEN 'POTASSIUM'
        WHEN itemid = 51275 THEN 'PTT'
        WHEN itemid = 51237 THEN 'INR'
        WHEN itemid = 51274 THEN 'PT'
        WHEN itemid = 50824 THEN 'SODIUM'
        WHEN itemid = 50983 THEN 'SODIUM'
        WHEN itemid = 51006 THEN 'BUN'
        WHEN itemid = 51300 THEN 'WBC'
        WHEN itemid = 51301 THEN 'WBC'
        --WHEN itemid = 50889 THEN 'CRP'
        WHEN itemid = 50924 THEN 'Ferritin'
        WHEN itemid = 50998 THEN 'Transferrin' 
        WHEN itemid = 50910 THEN 'CreatineKinase'
        WHEN itemid = 50911 THEN  'CK_MB'
        WHEN itemid = 50915 THEN  'D_Dimer'
        WHEN itemid = 50963 THEN  'NTproBNP'
        WHEN itemid = 51288 THEN  'SedimentationRate'
        WHEN itemid = 51214 THEN  'Fibrinogen'
        WHEN itemid = 50954 THEN  'LDH'
        WHEN itemid = 50960 THEN  'Magnesium'
        WHEN itemid = 50808 THEN 'Calcium_free'
        WHEN itemid = 50821 THEN 'pO2_bloodgas'
        WHEN itemid = 50820 THEN 'pH_bloodgas'
        WHEN itemid = 50818 THEN  'pCO2_bloodgas'
        WHEN itemid = 50817 THEN 'SO2_bloodgas'
        WHEN itemid = 51003 THEN  'Troponin_T'
        WHEN itemid = 51014 THEN  'Glucose_CSF'
        WHEN itemid = 51024 THEN  'Total_Protein_Joint_Fluid'
        WHEN itemid = 51059 THEN 'Total_Protein_Pleural'
        WHEN itemid = 51070 THEN 'Urine_Albumin_Creatinine_ratio'
        WHEN itemid = 51128 THEN 'WBC_Ascites'

      ELSE null
    END AS label
  , -- add in some sanity checks on the values
  -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
    CASE
      WHEN itemid = 50862 and valuenum >    10 THEN null -- g/dL 'ALBUMIN'
      WHEN itemid = 50868 and valuenum > 10000 THEN null -- mEq/L 'ANION GAP'
      WHEN itemid = 51144 and valuenum <     0 THEN null -- immature band forms, %
      WHEN itemid = 51144 and valuenum >   100 THEN null -- immature band forms, %
      WHEN itemid = 50882 and valuenum > 10000 THEN null -- mEq/L 'BICARBONATE'
      WHEN itemid = 50885 and valuenum >   150 THEN null -- mg/dL 'BILIRUBIN'
      WHEN itemid = 50806 and valuenum > 10000 THEN null -- mEq/L 'CHLORIDE'
      WHEN itemid = 50902 and valuenum > 10000 THEN null -- mEq/L 'CHLORIDE'
      WHEN itemid = 50912 and valuenum >   150 THEN null -- mg/dL 'CREATININE'
      WHEN itemid = 50809 and valuenum > 10000 THEN null -- mg/dL 'GLUCOSE'
      WHEN itemid = 50931 and valuenum > 10000 THEN null -- mg/dL 'GLUCOSE'
      WHEN itemid = 50810 and valuenum >   100 THEN null -- % 'HEMATOCRIT'
      WHEN itemid = 51221 and valuenum >   100 THEN null -- % 'HEMATOCRIT'
      WHEN itemid = 50811 and valuenum >    50 THEN null -- g/dL 'HEMOGLOBIN'
      WHEN itemid = 51222 and valuenum >    50 THEN null -- g/dL 'HEMOGLOBIN'
      WHEN itemid = 50813 and valuenum >    50 THEN null -- mmol/L 'LACTATE'
      WHEN itemid = 51265 and valuenum > 10000 THEN null -- K/uL 'PLATELET'
      WHEN itemid = 50822 and valuenum >    30 THEN null -- mEq/L 'POTASSIUM'
      WHEN itemid = 50971 and valuenum >    30 THEN null -- mEq/L 'POTASSIUM'
      WHEN itemid = 51275 and valuenum >   150 THEN null -- sec 'PTT'
      WHEN itemid = 51237 and valuenum >    50 THEN null -- 'INR'
      WHEN itemid = 51274 and valuenum >   150 THEN null -- sec 'PT'
      WHEN itemid = 50824 and valuenum >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
      WHEN itemid = 50983 and valuenum >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
      WHEN itemid = 51006 and valuenum >   300 THEN null -- 'BUN'
      WHEN itemid = 51300 and valuenum >  1000 THEN null -- 'WBC'
      WHEN itemid = 51301 and valuenum >  1000 THEN null -- 'WBC'

      WHEN itemid = 50861 and valuenum >  10000 THEN null -- IU/L,  Alanine Aminotransferase (ALT)  Blood Chemistry
      WHEN itemid = 50878 and valuenum >  10000 THEN null -- IU/L, Asparate Aminotransferase (AST) Blood Chemistry
      WHEN itemid = 50866 and valuenum >  10000 THEN null -- umol/L Ammonia Blood Chemistry
 --     WHEN itemid = 50889 and valuenum >  10000 THEN null -- mg/dl or mg/L! C-Reactive Protein  Blood Chemistry
      WHEN itemid = 50924 and valuenum > 10000 THEN null -- ng/mL Ferritin  Blood Chemistry
      WHEN itemid = 50998 and valuenum > 10000 THEN null -- mg/dl Transferrin Blood Chemistry
      WHEN itemid = 50910 and valuenum > 20000 THEN null -- IU/L Creatine Kinase (CK)  Blood Chemistry
      WHEN itemid = 50911 and valuenum > 10000 THEN null -- ng/mL Creatine Kinase, MB Isoenzyme Blood Chemistry
      WHEN itemid = 50915 and valuenum > 100000 THEN null  -- ng/mL D-Dimer Blood Chemistry
      WHEN itemid = 50963 and valuenum > 100000 THEN null  -- pg/mL NTproBNP  Blood Chemistry
      WHEN itemid = 51288 and valuenum > 5000 THEN null  -- mm/hr Sedimentation Rate  Blood Hematology
      WHEN itemid = 51214 and valuenum > 10000 THEN null -- mg/dL Fibrinogen, Functional  Blood Hematology
      WHEN itemid = 50954 and valuenum > 10000 THEN null -- IU/L Lactate Dehydrogenase (LD)  Blood Chemistry
      WHEN itemid = 50960 and valuenum > 50 THEN null -- mg/dL Magnesium Blood Chemistry
      WHEN itemid = 50808 and valuenum > 50 THEN null -- mmol/L Free Calcium  Blood Blood Gas
      WHEN itemid = 50821 and valuenum > 1000 THEN null -- mm Hg, pO2 Blood Blood Gas
      WHEN itemid = 50820 and valuenum > 15 THEN null -- pH units Blood Blood Gas
      WHEN itemid = 50818 and valuenum > 500 THEN null  -- mm Hg, pCO2  Blood Blood Gas
      WHEN itemid = 50817 and valuenum > 100 THEN null  -- %, Oxygen Saturation Blood Gas --> SpO2_bloodgas
      WHEN itemid = 51003 and valuenum > 100 THEN null  -- ng/mL, Troponin T  Blood Chemistry
      WHEN itemid = 51014 and valuenum > 10000 THEN null  -- mg/dL Glucose, CSF  Cerebrospinal Fluid (CSF) Chemistry
      WHEN itemid = 51024 and valuenum > 100 THEN null  -- g/dL, Total Protein, Joint Fluid  Joint Fluid Chemistry
      WHEN itemid = 51059 and valuenum > 100 THEN null  -- g/dL, Total Protein, Pleural  Pleural Chemistry
      WHEN itemid = 51070 and valuenum > 100000 THEN null  -- mg/g Albumin/Creatinine, Urine Urine Chemistry
      WHEN itemid = 51128 and valuenum > 100000 THEN null  -- #/uL (= /cubic mm) WBC, Ascites  Ascites Hematology

    ELSE le.valuenum
    END AS valuenum

  from matched_controls_hourly ch
      left join icustays ie
      on ch.icustay_id = ie.icustay_id
    LEFT JOIN labevents le
      ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id
      AND le.charttime BETWEEN (ch.control_onset_time - interval '48' hour) AND (ch.control_onset_time+interval '7' hour)
      AND le.ITEMID in
    (
      -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
      50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
      50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
      51144, -- BANDS - hematology
      50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
      50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
      50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
      50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
      50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
      50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
      50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
      51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
      50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
      51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
      50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
      50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
      51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
      50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
      50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
      51275, -- PTT | HEMATOLOGY | BLOOD | 474937
      51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
      51274, -- PT | HEMATOLOGY | BLOOD | 469090
      50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
      50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
      51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
      51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
      51300,  -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
      -- ADDED: 
      50861,  -- Alanine Aminotransferase (ALT)  Blood Chemistry
      50878,  -- Asparate Aminotransferase (AST) Blood Chemistry
      50866,  -- Ammonia Blood Chemistry
      --50889,  -- C-Reactive Protein  Blood Chemistry
      50924,  -- Ferritin  Blood Chemistry
      50998,  -- Transferrin Blood Chemistry
      50910,  -- Creatine Kinase (CK)  Blood Chemistry
      50911,  -- Creatine Kinase, MB Isoenzyme Blood Chemistry
      50915,  -- D-Dimer Blood Chemistry
      50963,  -- NTproBNP  Blood Chemistry
      51288,  -- Sedimentation Rate  Blood Hematology
      51214,  -- Fibrinogen, Functional  Blood Hematology
      50954,  -- Lactate Dehydrogenase (LD)  Blood Chemistry
      50960,  -- Magnesium Blood Chemistry
      50808,  -- Free Calcium  Blood Blood Gas
      50821,  -- pO2 Blood Blood Gas
      50820,  -- pH  Blood Blood Gas
      50818,  -- pCO2  Blood Blood Gas
      50817,  -- Oxygen Saturation Blood Blood Gas --> SpO2_bloodgas
      51003,  -- Troponin T  Blood Chemistry
      51014,  -- Glucose, CSF  Cerebrospinal Fluid (CSF) Chemistry
      51024,  -- Total Protein, Joint Fluid  Joint Fluid Chemistry
      51059,  -- Total Protein, Pleural  Pleural Chemistry
      51070,  -- Albumin/Creatinine, Urine Urine Chemistry
      51128  -- WBC, Ascites  Ascites Hematology

    )
    AND valuenum IS NOT null AND valuenum > 0 -- lab values cannot be 0 and cannot be negative
) pvt
--GROUP BY pvt.icustay_id, pvt.subject_id, pvt.hadm_id 
ORDER BY pvt.icustay_id, pvt.subject_id, pvt.hadm_id, pvt.chart_time;






