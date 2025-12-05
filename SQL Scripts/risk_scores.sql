create table risk_scores
(
    student_id   INTEGER,
    student_name TEXT not null,
    module       TEXT not null,
    risk_score   REAL,
    foreign key (student_id) references students (student_id),
    primary key (student_id, module)
);
