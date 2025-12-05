create table grades
(
    student_id           INTEGER,
    student_name         TEXT    not null,
    module               TEXT    not null,
    assessment_number    INTEGER not null,
    score                REAL    not null,
    progress_in_semester REAL not null,
    primary key (student_id, module, assessment_number)
);
