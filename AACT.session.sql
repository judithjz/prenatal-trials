SELECT locations.country, COUNT(studies.nct_id) AS active_trials
    FROM studies
    JOIN locations ON studies.nct_id = locations.nct_id
    WHERE studies.overall_status = 'Recruiting'
    GROUP BY locations.country
    ORDER BY active_trials DESC
    LIMIT 10;