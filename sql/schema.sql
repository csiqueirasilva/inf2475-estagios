PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE companies (
  contratante_id               INTEGER PRIMARY KEY,
  nome_contratante             TEXT,
  eh_pf                        BOOLEAN,
  cnpj_invalido                BOOLEAN,
  municipio                    TEXT,
  uf                           TEXT,
  bairro                       TEXT,
  nome_fantasia                TEXT,
  nome_receita                 TEXT,
  abertura                     TEXT,
  situacao                     TEXT,
  tipo                         TEXT,
  dt_atualizacao               TEXT,
  porte                        TEXT,
  capital_social               REAL,
  atividade_principal          TEXT,
  atividades_secundarias       TEXT
);

CREATE TABLE students (
  fonte_aluno      TEXT,
  matricula        TEXT,
  course_name      TEXT,
  PRIMARY KEY (fonte_aluno, matricula)
);

ALTER TABLE students ADD COLUMN name TEXT;
ALTER TABLE students ADD COLUMN birthdate TEXT;
ALTER TABLE students ADD COLUMN gender TEXT;
ALTER TABLE students ADD COLUMN cpf TEXT;
ALTER TABLE students ADD COLUMN email TEXT;
ALTER TABLE students ADD COLUMN tel TEXT;
ALTER TABLE students ADD COLUMN cel TEXT;
ALTER TABLE students ADD COLUMN address TEXT;
ALTER TABLE students ADD COLUMN city TEXT;
ALTER TABLE students ADD COLUMN state TEXT;
ALTER TABLE students ADD COLUMN areas_de_interesse TEXT;
ALTER TABLE students ADD COLUMN educations TEXT;
ALTER TABLE students ADD COLUMN experiences TEXT;
ALTER TABLE students ADD COLUMN languages TEXT;

CREATE TABLE internships (
  contract_id      INTEGER PRIMARY KEY,
  fonte_aluno      TEXT,
  matricula        TEXT NOT NULL,
  contratante_id   INTEGER NOT NULL,
  dt_cadastro      DATETIME,
  dt_rescisao      DATE,
  remuneracao      REAL,
  FOREIGN KEY(fonte_aluno, matricula)      REFERENCES students(fonte_aluno, matricula),
  FOREIGN KEY(contratante_id) REFERENCES companies(contratante_id)
);

CREATE TABLE actions (
  action_id      INTEGER PRIMARY KEY,
  log_action     TEXT,
  dt_log         DATETIME,
  tipo_usuario   TEXT,
  matricula      TEXT,
  job_id         INTEGER,
  offer_type     TEXT,
  fonte          TEXT,
  FOREIGN KEY(matricula) REFERENCES students(matricula),
  FOREIGN KEY(job_id)    REFERENCES job_offers(job_id)
);

CREATE TABLE job_offers (
  job_id          INTEGER PRIMARY KEY,
  contratante_id  INTEGER,
  date_time       DATETIME,
  requirements    TEXT,
  activities      TEXT,
  period_beg      INTEGER,
  period_end      INTEGER,
  city            TEXT,
  neighbourhood   TEXT,
  course_names    TEXT,
  FOREIGN KEY(contratante_id) REFERENCES companies(contratante_id)
);

CREATE TABLE raw_courses (
  id   INTEGER PRIMARY KEY,
  nome TEXT
);

CREATE TABLE raw_internships (
  contract_id                             INTEGER PRIMARY KEY,
  id_contratante                          INTEGER,
  dt_cadastro                             TEXT,  -- DATETIME string
  dt_inicio                               TEXT,  -- DATE string
  dt_termino                              TEXT,
  vale_transporte                         BOOLEAN,
  data_rescisao                           TEXT,
  horarioflexivel                         BOOLEAN,
  trabalhoremoto                          BOOLEAN,
  horas_semanais                          INTEGER,
  valor_vale_transporte                   REAL,
  remuneracao                             REAL,
  fonte_aluno                             TEXT,
  matricula                               TEXT,
  curso_nivel                             TEXT,
  curso_nome                              TEXT,
  curso_habilitacao                       TEXT,
  curso_enfase                            TEXT,
  curso_apresentacao                      TEXT,
  pessoa_id                               INTEGER,
  nome_contratante                        TEXT,
  eh_pf                                   BOOLEAN,
  cnpj_invalido                           BOOLEAN,
  nome_fantasia_receita                   TEXT,
  nome_receita                            TEXT,
  abertura_receita                        TEXT,
  municipio_receita                       TEXT,
  uf_receita                              TEXT,
  bairro_receita                          TEXT,
  situacao_receita                        TEXT,
  tipo_receita                            TEXT,
  dt_atualizacao_receita                  TEXT,
  porte_receita                           TEXT,
  capital_social_receita                  REAL,
  contratante_atividade_principal         TEXT,
  qtd_estagios_do_aluno_nesse_contratante INTEGER,
  dias_de_estagio                         INTEGER,
  contratante_atividades_secundarias      TEXT,
  qtd_documentos                          INTEGER
);

CREATE TABLE raw_offers_and_logs (
  log_action        TEXT,
  dt_log_action     TEXT,
  tipo_usuario_fonte TEXT,
  tipo_usuario_alvo  TEXT,
  id_usuario_fonte  INTEGER,
  id_usuario_alvo   INTEGER,
  fonte             TEXT,
  matricula         TEXT,
  idcurso           TEXT,
  resultados        TEXT,
  offer             TEXT,
  id                INTEGER PRIMARY KEY,
  date_time         TEXT,
  type              TEXT,
  office_hours      TEXT,
  remuneration      REAL,
  periodo_beg       TEXT,
  periodo_end       TEXT,
  gender            TEXT,
  job_openings      INTEGER,
  benefits          TEXT,
  requirements      TEXT,
  activities        TEXT,
  city              TEXT,
  neighbourhood     TEXT,
  procedures        TEXT,
  enterprise_id     INTEGER,
  enterprise_name   TEXT,
  data_termino      TEXT,
  external_link     TEXT,
  cursos_da_oferta  TEXT,
  FOREIGN KEY(idcurso) REFERENCES raw_courses(id)
);

CREATE TABLE raw_student_course_codes (
  matricula     TEXT PRIMARY KEY,
  sigla_curso   TEXT,
  sigla_hab     TEXT,
  sigla_enf     TEXT
);

CREATE TABLE raw_course_code_map (
  sigla_curso   TEXT,
  sigla_hab     TEXT,
  sigla_enf     TEXT,
  nome          TEXT,
  id   INTEGER PRIMARY KEY
);

-- === COMPANIES ===
CREATE INDEX IF NOT EXISTS idx_companies_nome_contratante ON companies(nome_contratante);
CREATE INDEX IF NOT EXISTS idx_companies_municipio ON companies(municipio);
CREATE INDEX IF NOT EXISTS idx_companies_uf ON companies(uf);
CREATE INDEX IF NOT EXISTS idx_companies_situacao ON companies(situacao);
CREATE INDEX IF NOT EXISTS idx_companies_tipo ON companies(tipo);
CREATE INDEX IF NOT EXISTS idx_companies_nome_receita ON companies(nome_receita);

-- === STUDENTS ===
-- Primary key already covers (fonte_aluno, matricula)
CREATE INDEX IF NOT EXISTS idx_students_course_name ON students(course_name);
CREATE INDEX IF NOT EXISTS idx_students_fonte_matricula ON students(fonte_aluno, matricula);

-- === INTERNSHIPS ===
CREATE INDEX IF NOT EXISTS idx_internships_fonte_aluno ON internships(fonte_aluno);
CREATE INDEX IF NOT EXISTS idx_internships_matricula ON internships(matricula);
CREATE INDEX IF NOT EXISTS idx_internships_contratante_id ON internships(contratante_id);
CREATE INDEX IF NOT EXISTS idx_internships_dt_cadastro ON internships(dt_cadastro);
CREATE INDEX IF NOT EXISTS idx_internships_dt_rescisao ON internships(dt_rescisao);

-- === JOB_OFFERS ===
CREATE INDEX IF NOT EXISTS idx_offers_contratante_id ON job_offers(contratante_id);
CREATE INDEX IF NOT EXISTS idx_offers_date_time ON job_offers(date_time);
CREATE INDEX IF NOT EXISTS idx_offers_city ON job_offers(city);
CREATE INDEX IF NOT EXISTS idx_offers_course_names ON job_offers(course_names);

-- === ACTIONS ===
CREATE INDEX IF NOT EXISTS idx_actions_matricula ON actions(matricula);
CREATE INDEX IF NOT EXISTS idx_actions_job_id ON actions(job_id);
CREATE INDEX IF NOT EXISTS idx_actions_tipo_usuario ON actions(tipo_usuario);
CREATE INDEX IF NOT EXISTS idx_actions_dt_log ON actions(dt_log);

-- === RAW_COURSES ===
CREATE INDEX IF NOT EXISTS idx_raw_courses_nome ON raw_courses(nome);

-- === RAW_INTERNSHIPS ===
CREATE INDEX IF NOT EXISTS idx_rawinternships_id_contratante ON raw_internships(id_contratante);
CREATE INDEX IF NOT EXISTS idx_rawinternships_nome_contratante ON raw_internships(nome_contratante);
CREATE INDEX IF NOT EXISTS idx_rawinternships_municipio_receita ON raw_internships(municipio_receita);
CREATE INDEX IF NOT EXISTS idx_rawinternships_uf_receita ON raw_internships(uf_receita);
CREATE INDEX IF NOT EXISTS idx_rawinternships_pessoa_id ON raw_internships(pessoa_id);
CREATE INDEX IF NOT EXISTS idx_rawinternships_fonte_aluno ON raw_internships(fonte_aluno);
CREATE INDEX IF NOT EXISTS idx_rawinternships_matricula ON raw_internships(matricula);

-- === RAW_OFFERS_AND_LOGS ===
CREATE INDEX IF NOT EXISTS idx_rawlogs_matricula ON raw_offers_and_logs(matricula);
CREATE INDEX IF NOT EXISTS idx_rawlogs_enterprise_id ON raw_offers_and_logs(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_rawlogs_idcurso ON raw_offers_and_logs(idcurso);
CREATE INDEX IF NOT EXISTS idx_rawlogs_dt_log_action ON raw_offers_and_logs(dt_log_action);
CREATE INDEX IF NOT EXISTS idx_rawlogs_date_time ON raw_offers_and_logs(date_time);
CREATE INDEX IF NOT EXISTS idx_rawlogs_tipo_usuario_fonte ON raw_offers_and_logs(tipo_usuario_fonte);