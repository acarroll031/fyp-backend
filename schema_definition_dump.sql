--
-- PostgreSQL database dump
--

\restrict dEEuEeB6b429g2D0WycMYEL5vPIiz2sflejU0LJG8l7wbVr1R5sbhSxs5yNlgL4

-- Dumped from database version 17.8 (6108b59)
-- Dumped by pg_dump version 17.7

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: grades; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.grades (
    student_id integer NOT NULL,
    student_name text NOT NULL,
    module text NOT NULL,
    assessment_number integer NOT NULL,
    score real NOT NULL,
    progress_in_semester real NOT NULL
);


--
-- Name: lecturers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.lecturers (
    email text NOT NULL,
    lecturer_name text NOT NULL,
    password_hash text NOT NULL
);


--
-- Name: modules; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.modules (
    module_code text NOT NULL,
    module_name text NOT NULL,
    lecturer_email text,
    assessment_count integer NOT NULL
);


--
-- Name: notifications; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.notifications (
    id integer NOT NULL,
    lecturer_email character varying(255) NOT NULL,
    message text NOT NULL,
    is_read boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    notification_type character varying(50),
    module text
);


--
-- Name: notifications_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.notifications_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: notifications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.notifications_id_seq OWNED BY public.notifications.id;


--
-- Name: risk_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.risk_history (
    id integer NOT NULL,
    student_id integer,
    student_name character varying(255),
    module character varying(50),
    risk_score double precision,
    recorded_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: risk_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.risk_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: risk_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.risk_history_id_seq OWNED BY public.risk_history.id;


--
-- Name: risk_scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.risk_scores (
    student_id integer NOT NULL,
    student_name text NOT NULL,
    module text NOT NULL,
    risk_score real,
    previous_risk_score real
);


--
-- Name: students; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.students (
    student_id integer NOT NULL,
    student_name text NOT NULL,
    module text NOT NULL,
    average_score real,
    assessments_completed integer,
    performance_trend real,
    max_consecutive_misses integer,
    progress_in_semester double precision,
    email character varying(255)
);


--
-- Name: notifications id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.notifications ALTER COLUMN id SET DEFAULT nextval('public.notifications_id_seq'::regclass);


--
-- Name: risk_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.risk_history ALTER COLUMN id SET DEFAULT nextval('public.risk_history_id_seq'::regclass);


--
-- Name: grades grades_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.grades
    ADD CONSTRAINT grades_pkey PRIMARY KEY (student_id, module, assessment_number);


--
-- Name: lecturers lecturers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lecturers
    ADD CONSTRAINT lecturers_pkey PRIMARY KEY (email);


--
-- Name: modules modules_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.modules
    ADD CONSTRAINT modules_pkey PRIMARY KEY (module_code);


--
-- Name: notifications notifications_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.notifications
    ADD CONSTRAINT notifications_pkey PRIMARY KEY (id);


--
-- Name: risk_history risk_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.risk_history
    ADD CONSTRAINT risk_history_pkey PRIMARY KEY (id);


--
-- Name: risk_scores risk_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.risk_scores
    ADD CONSTRAINT risk_scores_pkey PRIMARY KEY (student_id, module);


--
-- Name: students students_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_pkey PRIMARY KEY (student_id, module);


--
-- Name: students students_student_id_module_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_student_id_module_unique UNIQUE (student_id, module);


--
-- Name: grades constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.grades
    ADD CONSTRAINT constraint_1 FOREIGN KEY (student_id, module) REFERENCES public.students(student_id, module) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: notifications constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.notifications
    ADD CONSTRAINT constraint_1 FOREIGN KEY (module) REFERENCES public.modules(module_code) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: risk_history constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.risk_history
    ADD CONSTRAINT constraint_1 FOREIGN KEY (student_id, module) REFERENCES public.students(student_id, module) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: students constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT constraint_1 FOREIGN KEY (module) REFERENCES public.modules(module_code) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: modules modules_lecturer_email_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.modules
    ADD CONSTRAINT modules_lecturer_email_fkey FOREIGN KEY (lecturer_email) REFERENCES public.lecturers(email) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: risk_scores risk_scores_student_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.risk_scores
    ADD CONSTRAINT risk_scores_student_fk FOREIGN KEY (student_id, module) REFERENCES public.students(student_id, module) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict dEEuEeB6b429g2D0WycMYEL5vPIiz2sflejU0LJG8l7wbVr1R5sbhSxs5yNlgL4

