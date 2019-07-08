/*
Looping through the 72 hour window of suspicion of infection to determine SOFA score on a hourly base
---------------------------------------------------------------------------------------------------------------------
- AUTHOR (of this version): Michael Moor, February 2018
---------------------------------------------------------------------------------------------------------------------
*/

DROP FUNCTION IF EXISTS sofa_loop();
CREATE OR REPLACE FUNCTION sofa_loop() RETURNS void 
	AS
	$$
	BEGIN
        FOR i IN  0..72 LOOP -- try out only one loop iteration!
        	raise notice 'Loop index: %', i;
            PERFORM get_uo(i)
            , get_vitals(i)
            , get_gcs(i)
            , get_labs(i)
            , get_bg(i)
            , get_vaso_mv(i)
            , get_pafi1(i)
            , get_pafi2(i)
            , score_compute(i);
        END LOOP;
    END;
	$$
	LANGUAGE plpgsql;
