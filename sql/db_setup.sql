CREATE DATABASE complaints_db;

\connect complaints_db;

CREATE TABLE complaints(
    unique_mos_id INT,
    first_name VARCHAR,
    last_name VARCHAR,
    command_now VARCHAR,
    shield_no INT,
    complaint_id INT,
    month_received INT,
    year_received INT,
    month_closed INT,
    year_closed INT,
    command_at_incident VARCHAR,
    rank_abbrev_incident VARCHAR,
    rank_abbrev_now VARCHAR,
    rank_now VARCHAR,
    rank_incident VARCHAR,
    mos_ethnicity VARCHAR,
    mos_gender VARCHAR,
    mos_age_incident INT,
    complainant_ethnicity VARCHAR,
    complainant_gender VARCHAR,
    complainant_age_incident INT,
    fado_type VARCHAR,
    allegation VARCHAR,
    precinct INT,
    contact_reason VARCHAR,
    outcome_description VARCHAR,
    board_disposition VARCHAR
);

\copy complaints FROM 'allegations.csv' DELIMITER ',' CSV HEADER;