create table lecturers
(
    email         TEXT
        primary key,
    lecturer_name TEXT not null,
    password_hash TEXT not null
);

INSERT INTO lecturers (email, lecturer_name, password_hash) VALUES ('lecturer@test.com', 'Adam Carroll', '$2b$12$2rca0GQ1pWAF5X588kurc.t2nVUhFQj1BC8XNKjFxAwublILQB7xO');
