-- requires postgresql with pgvector extension enabled

CREATE TABLE public.cv_embeddings (
	fonte_aluno text NOT NULL,
	matricula text NOT NULL,
	raw_input text NOT NULL,
	llm_parsed_raw_input text NOT NULL,
	embedding public.vector(768) NULL,
	latent_code public.vector(96) NULL,
	last_update timestamptz DEFAULT now() NOT NULL,
	raw_embedding public.vector(768) NULL,
	CONSTRAINT cv_embeddings_pkey PRIMARY KEY (fonte_aluno, matricula)
);

CREATE TABLE public.job_embeddings (
	fonte_aluno text NOT NULL,
	matricula text NOT NULL,
	contract_id int4 NOT NULL,
	raw_input text NOT NULL,
	embedding public.vector(768) NULL,
	latent_code public.vector(96) NULL,
	last_update timestamptz DEFAULT now() NOT NULL,
	CONSTRAINT job_embeddings_pkey PRIMARY KEY (fonte_aluno, matricula, contract_id)
);

CREATE TABLE public.job_clusters (
	fonte_aluno text NOT NULL,
	matricula text NOT NULL,
	contract_id int4 NOT NULL,
	cluster_id int4 NOT NULL,
	CONSTRAINT job_clusters_pkey PRIMARY KEY (fonte_aluno, matricula, contract_id)
);

CREATE TABLE public.job_cluster_centroids (
	cluster_id int4 NOT NULL,
	centroid public.vector(768) NOT NULL,
	CONSTRAINT job_cluster_centroids_pkey PRIMARY KEY (cluster_id)
);

CREATE TABLE public.cv_clusters (
	fonte_aluno text NOT NULL,
	matricula text NOT NULL,
	cluster_id int4 NOT NULL,
	CONSTRAINT cv_clusters_pkey PRIMARY KEY (fonte_aluno, matricula)
);

CREATE TABLE public.cv_cluster_centroids (
	cluster_id int4 NOT NULL,
	centroid public.vector(768) NOT NULL,
	CONSTRAINT cv_cluster_centroids_pkey PRIMARY KEY (cluster_id)
);

CREATE TABLE public.cv_nomic_clusters (
	fonte_aluno text NOT NULL,
	matricula text NOT NULL,
	cluster_id int4 NOT NULL,
	CONSTRAINT cv_nomic_clusters_pkey PRIMARY KEY (fonte_aluno, matricula)
);

CREATE TABLE public.cv_nomic_centroids (
	cluster_id int4 NOT NULL,
	centroid public.vector(768) NOT NULL,
	CONSTRAINT cv_nomic_centroids_pkey PRIMARY KEY (cluster_id)
);
