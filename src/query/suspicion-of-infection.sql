/*
- ORIGINAL VERSION. 
- SOURCE: https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/suspicion-of-infection.sql
- DOWNLOADED on 8th February 2018
*/

DROP TABLE IF EXISTS suspinfect_poe CASCADE;
CREATE TABLE suspinfect_poe as
with abx as
(
  select icustay_id
    , suspected_infection_time
    , specimen, positiveculture
    , si_starttime, si_endtime
    , antibiotic_name
    , antibiotic_time
    , ROW_NUMBER() OVER
    (
      PARTITION BY icustay_id
      ORDER BY suspected_infection_time
    ) as rn
  from abx_micro_poe
)
select
  ie.icustay_id
  , antibiotic_name
  , antibiotic_time
  , suspected_infection_time
  , specimen, positiveculture
  , si_starttime, si_endtime
from icustays ie
left join abx
  on ie.icustay_id = abx.icustay_id
  and abx.rn = 1
order by ie.icustay_id;
