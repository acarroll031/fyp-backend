--
-- PostgreSQL database dump
--

\restrict tn8IJwPLx1eBEcZ2TWsMZmh7U3HueoVV6pxfWqycHLGBp3AdfGsQG8oMcKixiNc

-- Dumped from database version 17.8 (6108b59)
-- Dumped by pg_dump version 17.7

-- Started on 2026-02-25 18:46:49

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
-- TOC entry 220 (class 1259 OID 49486)
-- Name: grades; Type: TABLE; Schema: public; Owner: neondb_owner
--

CREATE TABLE public.grades (
    student_id integer NOT NULL,
    student_name text NOT NULL,
    module text NOT NULL,
    assessment_number integer NOT NULL,
    score real NOT NULL,
    progress_in_semester real NOT NULL
);


ALTER TABLE public.grades OWNER TO neondb_owner;

--
-- TOC entry 217 (class 1259 OID 49455)
-- Name: lecturers; Type: TABLE; Schema: public; Owner: neondb_owner
--

CREATE TABLE public.lecturers (
    email text NOT NULL,
    lecturer_name text NOT NULL,
    password_hash text NOT NULL
);


ALTER TABLE public.lecturers OWNER TO neondb_owner;

--
-- TOC entry 218 (class 1259 OID 49462)
-- Name: modules; Type: TABLE; Schema: public; Owner: neondb_owner
--

CREATE TABLE public.modules (
    module_code text NOT NULL,
    module_name text NOT NULL,
    lecturer_email text,
    assessment_count integer NOT NULL
);


ALTER TABLE public.modules OWNER TO neondb_owner;

--
-- TOC entry 225 (class 1259 OID 122881)
-- Name: notifications; Type: TABLE; Schema: public; Owner: neondb_owner
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


ALTER TABLE public.notifications OWNER TO neondb_owner;

--
-- TOC entry 224 (class 1259 OID 122880)
-- Name: notifications_id_seq; Type: SEQUENCE; Schema: public; Owner: neondb_owner
--

CREATE SEQUENCE public.notifications_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.notifications_id_seq OWNER TO neondb_owner;

--
-- TOC entry 3389 (class 0 OID 0)
-- Dependencies: 224
-- Name: notifications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: neondb_owner
--

ALTER SEQUENCE public.notifications_id_seq OWNED BY public.notifications.id;


--
-- TOC entry 223 (class 1259 OID 81921)
-- Name: risk_history; Type: TABLE; Schema: public; Owner: neondb_owner
--

CREATE TABLE public.risk_history (
    id integer NOT NULL,
    student_id integer,
    student_name character varying(255),
    module character varying(50),
    risk_score double precision,
    recorded_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.risk_history OWNER TO neondb_owner;

--
-- TOC entry 222 (class 1259 OID 81920)
-- Name: risk_history_id_seq; Type: SEQUENCE; Schema: public; Owner: neondb_owner
--

CREATE SEQUENCE public.risk_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.risk_history_id_seq OWNER TO neondb_owner;

--
-- TOC entry 3390 (class 0 OID 0)
-- Dependencies: 222
-- Name: risk_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: neondb_owner
--

ALTER SEQUENCE public.risk_history_id_seq OWNED BY public.risk_history.id;


--
-- TOC entry 221 (class 1259 OID 49493)
-- Name: risk_scores; Type: TABLE; Schema: public; Owner: neondb_owner
--

CREATE TABLE public.risk_scores (
    student_id integer NOT NULL,
    student_name text NOT NULL,
    module text NOT NULL,
    risk_score real,
    previous_risk_score real
);


ALTER TABLE public.risk_scores OWNER TO neondb_owner;

--
-- TOC entry 219 (class 1259 OID 49474)
-- Name: students; Type: TABLE; Schema: public; Owner: neondb_owner
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


ALTER TABLE public.students OWNER TO neondb_owner;

--
-- TOC entry 3214 (class 2604 OID 122884)
-- Name: notifications id; Type: DEFAULT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.notifications ALTER COLUMN id SET DEFAULT nextval('public.notifications_id_seq'::regclass);


--
-- TOC entry 3212 (class 2604 OID 81924)
-- Name: risk_history id; Type: DEFAULT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.risk_history ALTER COLUMN id SET DEFAULT nextval('public.risk_history_id_seq'::regclass);


--
-- TOC entry 3226 (class 2606 OID 49492)
-- Name: grades grades_pkey; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.grades
    ADD CONSTRAINT grades_pkey PRIMARY KEY (student_id, module, assessment_number);


--
-- TOC entry 3218 (class 2606 OID 49461)
-- Name: lecturers lecturers_pkey; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.lecturers
    ADD CONSTRAINT lecturers_pkey PRIMARY KEY (email);


--
-- TOC entry 3220 (class 2606 OID 49468)
-- Name: modules modules_pkey; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.modules
    ADD CONSTRAINT modules_pkey PRIMARY KEY (module_code);


--
-- TOC entry 3232 (class 2606 OID 122890)
-- Name: notifications notifications_pkey; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.notifications
    ADD CONSTRAINT notifications_pkey PRIMARY KEY (id);


--
-- TOC entry 3230 (class 2606 OID 81927)
-- Name: risk_history risk_history_pkey; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.risk_history
    ADD CONSTRAINT risk_history_pkey PRIMARY KEY (id);


--
-- TOC entry 3228 (class 2606 OID 49499)
-- Name: risk_scores risk_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.risk_scores
    ADD CONSTRAINT risk_scores_pkey PRIMARY KEY (student_id, module);


--
-- TOC entry 3222 (class 2606 OID 73731)
-- Name: students students_pkey; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_pkey PRIMARY KEY (student_id, module);


--
-- TOC entry 3224 (class 2606 OID 73729)
-- Name: students students_student_id_module_unique; Type: CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_student_id_module_unique UNIQUE (student_id, module);


--
-- TOC entry 3235 (class 2606 OID 196625)
-- Name: grades constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.grades
    ADD CONSTRAINT constraint_1 FOREIGN KEY (student_id, module) REFERENCES public.students(student_id, module) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3238 (class 2606 OID 196688)
-- Name: notifications constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.notifications
    ADD CONSTRAINT constraint_1 FOREIGN KEY (module) REFERENCES public.modules(module_code) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3237 (class 2606 OID 196630)
-- Name: risk_history constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.risk_history
    ADD CONSTRAINT constraint_1 FOREIGN KEY (student_id, module) REFERENCES public.students(student_id, module) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3234 (class 2606 OID 196618)
-- Name: students constraint_1; Type: FK CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT constraint_1 FOREIGN KEY (module) REFERENCES public.modules(module_code) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3233 (class 2606 OID 196613)
-- Name: modules modules_lecturer_email_fkey; Type: FK CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.modules
    ADD CONSTRAINT modules_lecturer_email_fkey FOREIGN KEY (lecturer_email) REFERENCES public.lecturers(email) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 3236 (class 2606 OID 196635)
-- Name: risk_scores risk_scores_student_fk; Type: FK CONSTRAINT; Schema: public; Owner: neondb_owner
--

ALTER TABLE ONLY public.risk_scores
    ADD CONSTRAINT risk_scores_student_fk FOREIGN KEY (student_id, module) REFERENCES public.students(student_id, module) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- TOC entry 2070 (class 826 OID 16394)
-- Name: DEFAULT PRIVILEGES FOR SEQUENCES; Type: DEFAULT ACL; Schema: public; Owner: cloud_admin
--

ALTER DEFAULT PRIVILEGES FOR ROLE cloud_admin IN SCHEMA public GRANT ALL ON SEQUENCES TO neon_superuser WITH GRANT OPTION;


--
-- TOC entry 2069 (class 826 OID 16393)
-- Name: DEFAULT PRIVILEGES FOR TABLES; Type: DEFAULT ACL; Schema: public; Owner: cloud_admin
--

ALTER DEFAULT PRIVILEGES FOR ROLE cloud_admin IN SCHEMA public GRANT ALL ON TABLES TO neon_superuser WITH GRANT OPTION;


-- Completed on 2026-02-25 18:46:51

--
-- PostgreSQL database dump complete
--

\unrestrict tn8IJwPLx1eBEcZ2TWsMZmh7U3HueoVV6pxfWqycHLGBp3AdfGsQG8oMcKixiNc

