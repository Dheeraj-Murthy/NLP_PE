--
-- PostgreSQL database dump
--

\restrict YHq8K60QX3pVclOhomIvw4zrhgcp42Ppsk84mV4gy4guJyYg1LKMkQpXDAXyJLo

-- Dumped from database version 17.7 (Homebrew)
-- Dumped by pg_dump version 17.7 (Homebrew)

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

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: judgment_chunks; Type: TABLE; Schema: public; Owner: dheerajmurthy
--

CREATE TABLE public.judgment_chunks (
    chunk_id integer NOT NULL,
    judgment_id integer,
    section text,
    content text,
    CONSTRAINT judgment_chunks_section_check CHECK ((section = ANY (ARRAY['facts'::text, 'issues'::text, 'arguments'::text, 'ratio'::text, 'judgment'::text])))
);


ALTER TABLE public.judgment_chunks OWNER TO dheerajmurthy;

--
-- Name: judgment_chunks_chunk_id_seq; Type: SEQUENCE; Schema: public; Owner: dheerajmurthy
--

CREATE SEQUENCE public.judgment_chunks_chunk_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.judgment_chunks_chunk_id_seq OWNER TO dheerajmurthy;

--
-- Name: judgment_chunks_chunk_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: dheerajmurthy
--

ALTER SEQUENCE public.judgment_chunks_chunk_id_seq OWNED BY public.judgment_chunks.chunk_id;


--
-- Name: judgment_embeddings; Type: TABLE; Schema: public; Owner: dheerajmurthy
--

CREATE TABLE public.judgment_embeddings (
    chunk_id integer NOT NULL,
    embedding public.vector(1536)
);


ALTER TABLE public.judgment_embeddings OWNER TO dheerajmurthy;

--
-- Name: judgments; Type: TABLE; Schema: public; Owner: dheerajmurthy
--

CREATE TABLE public.judgments (
    id integer NOT NULL,
    petitioner text,
    respondent text,
    court text,
    date_of_judgment date,
    bench text[],
    citations jsonb,
    judgment_text text
);


ALTER TABLE public.judgments OWNER TO dheerajmurthy;

--
-- Name: judgments_id_seq; Type: SEQUENCE; Schema: public; Owner: dheerajmurthy
--

CREATE SEQUENCE public.judgments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.judgments_id_seq OWNER TO dheerajmurthy;

--
-- Name: judgments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: dheerajmurthy
--

ALTER SEQUENCE public.judgments_id_seq OWNED BY public.judgments.id;


--
-- Name: judgment_chunks chunk_id; Type: DEFAULT; Schema: public; Owner: dheerajmurthy
--

ALTER TABLE ONLY public.judgment_chunks ALTER COLUMN chunk_id SET DEFAULT nextval('public.judgment_chunks_chunk_id_seq'::regclass);


--
-- Name: judgments id; Type: DEFAULT; Schema: public; Owner: dheerajmurthy
--

ALTER TABLE ONLY public.judgments ALTER COLUMN id SET DEFAULT nextval('public.judgments_id_seq'::regclass);


--
-- Name: judgment_chunks judgment_chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: dheerajmurthy
--

ALTER TABLE ONLY public.judgment_chunks
    ADD CONSTRAINT judgment_chunks_pkey PRIMARY KEY (chunk_id);


--
-- Name: judgment_embeddings judgment_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: dheerajmurthy
--

ALTER TABLE ONLY public.judgment_embeddings
    ADD CONSTRAINT judgment_embeddings_pkey PRIMARY KEY (chunk_id);


--
-- Name: judgments judgments_pkey; Type: CONSTRAINT; Schema: public; Owner: dheerajmurthy
--

ALTER TABLE ONLY public.judgments
    ADD CONSTRAINT judgments_pkey PRIMARY KEY (id);


--
-- Name: judgment_embedding_hnsw_idx; Type: INDEX; Schema: public; Owner: dheerajmurthy
--

CREATE INDEX judgment_embedding_hnsw_idx ON public.judgment_embeddings USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: judgment_chunks judgment_chunks_judgment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: dheerajmurthy
--

ALTER TABLE ONLY public.judgment_chunks
    ADD CONSTRAINT judgment_chunks_judgment_id_fkey FOREIGN KEY (judgment_id) REFERENCES public.judgments(id) ON DELETE CASCADE;


--
-- Name: judgment_embeddings judgment_embeddings_chunk_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: dheerajmurthy
--

ALTER TABLE ONLY public.judgment_embeddings
    ADD CONSTRAINT judgment_embeddings_chunk_id_fkey FOREIGN KEY (chunk_id) REFERENCES public.judgment_chunks(chunk_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict YHq8K60QX3pVclOhomIvw4zrhgcp42Ppsk84mV4gy4guJyYg1LKMkQpXDAXyJLo

