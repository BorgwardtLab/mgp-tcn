
/*
Here, tables are created which will be subsequently 
filled with data required to compute the hourly sofa score

Author: Michael Moor, February 2018
*/



DROP TABLE IF EXISTS uo_table;
CREATE TABLE uo_table (
    icustay_id      int,
    UrineOutput     double precision,
    time_window     int
);

DROP TABLE IF EXISTS vitals_table;
CREATE TABLE vitals_table (
    icustay_id      int,
    MeanBP_Min      double precision,
    time_window     int
);

DROP TABLE IF EXISTS gcs_table;
CREATE TABLE gcs_table (
    icustay_id      int,
    MinGCS          double precision,
    time_window     int
);

DROP TABLE IF EXISTS labs_table;
CREATE TABLE labs_table (
    icustay_id      int,
    Bilirubin_Max   double precision,
    Creatinine_Max  double precision,
    Platelet_Min    double precision,
    time_window     int
);

DROP TABLE IF EXISTS bg_table;
CREATE TABLE bg_table (
    icustay_id      int,
    charttime       timestamp,
    PaO2FiO2        double precision,
    time_window     int
);


DROP TABLE IF EXISTS sofa_table;
CREATE TABLE sofa_table (
    icustay_id      int,
    sofa            int,
    time_window     int,
    respiration     int,
    coagulation     int, 
    liver           int,
    cardiovascular  int,
    cns             int,
    renal           int,
    cv_Mean_BP_u70  int,
    cv_vasopressor_usage int

);
    


-- only need metavision vaso: (not carevue)
DROP TABLE IF EXISTS vaso_mv_table;
CREATE TABLE vaso_mv_table (
    icustay_id              int,
    rate_norepinephrine     double precision,
    rate_epinephrine        double precision, 
    rate_dopamine           double precision,
    rate_dobutamine         double precision,
    time_window             int
);
 
DROP TABLE IF EXISTS pafi1_table;
CREATE TABLE pafi1_table (
    icustay_id              int,
    PaO2FiO2                double precision,
    IsVent                  int, 
    time_window             int
);


DROP TABLE IF EXISTS pafi2_table;
CREATE TABLE pafi2_table (
    icustay_id              int,
    PaO2FiO2_novent_min     double precision,
    PaO2FiO2_vent_min       double precision, 
    time_window             int
);

