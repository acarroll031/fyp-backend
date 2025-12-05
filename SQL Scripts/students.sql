create table students
(
    student_id             INTEGER
        primary key,
    student_name           TEXT not null,
    module                 TEXT not null
        references modules (module_code),
    average_score          REAL,
    assessments_completed  INTEGER,
    performance_trend      REAL,
    max_consecutive_misses INTEGER,
    progress_in_semester   FLOAT
);
