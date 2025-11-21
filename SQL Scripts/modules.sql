create table modules
(
    module_code      TEXT
        primary key,
    module_name      TEXT    not null,
    lecturer_email   INTEGER
        references lecturers,
    assessment_count INTEGER not null
);

INSERT INTO modules (module_code, module_name, lecturer_email, assessment_count) VALUES ('CS161', 'Introduction to Computer Science', 'lecturer@test.com', 12);
INSERT INTO modules (module_code, module_name, lecturer_email, assessment_count) VALUES ('CS210', 'Algorithms and Data Structures 1', 'lecturer@test.com', 8);
