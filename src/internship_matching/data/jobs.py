import ast
import json
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import psycopg2
import torch

from ..constants import JOBS_EMBEDDING_COLUMN_NAME
from ..data.db import POSTGRES_URL, SQLITE_URL
from tqdm import tqdm

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sanitize_input_jobs(sqlite_db_path: str = SQLITE_URL) -> List[Dict]:
    """
    Connect to SQLite, pull every internship/job,
    and build a list of dicts with keys:
       'fonte_aluno',
       'matricula',
       'contract_id',
       'nome_contratante',
       'localidade',
       'course_name',
       'atividade_principal',
       'atividades_secundarias'
    """
    conn = sqlite3.connect(sqlite_db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
    WITH RECURSIVE split_atividades AS (
  -- Anchor: seed each internship with its principais + the full secundárias list (ending in “;”)
  SELECT 
    i.rowid                                        AS rid,
    i.contract_id                                  AS contract_id,
    COALESCE(c.nome_fantasia, c.nome_contratante)  AS nome_contratante,
    c.uf || ' ' || c.municipio || ' ' || c.bairro  AS localidade,
    s.course_name                                 AS course_name,
    -- strip code from principal activity
    SUBSTR(
      c.atividade_principal,
      INSTR(c.atividade_principal, ' ')+1
    )                                              AS atividade_principal,
    ''                                             AS item,
    COALESCE(c.atividades_secundarias, '') || ';'  AS rest
  FROM internships i
  JOIN companies c 
    ON c.contratante_id = i.contratante_id
  JOIN students s 
    ON s.fonte_aluno = i.fonte_aluno
   AND s.matricula   = i.matricula
  WHERE 
    c.atividade_principal IS NOT NULL
    AND s.course_name <> 'DESCONHECIDO'

  UNION ALL

  -- Recursively peel off each “item;” from rest
  SELECT
    rid,
    contract_id,
    nome_contratante,
    localidade,
    course_name,
    atividade_principal,
    SUBSTR(rest, 1, INSTR(rest, ';')-1)            AS item,
    SUBSTR(rest, INSTR(rest, ';')+1)               AS rest
  FROM split_atividades
  WHERE rest <> ''
)
SELECT
  i.fonte_aluno,
  i.matricula,
  sa.contract_id,
  sa.nome_contratante,
  sa.localidade,
  sa.course_name,
  sa.atividade_principal,
  -- re-assemble all stripped items with real newlines
  GROUP_CONCAT(
    SUBSTR(sa.item, INSTR(sa.item, ' ')+1),
    CHAR(10)
  )                                              AS atividades_secundarias
FROM split_atividades sa
inner join internships i on i.contract_id = sa.contract_id
WHERE sa.item <> ''
GROUP BY
  sa.rid,
  sa.contract_id
    """)

    out: List[Dict] = []
    for row in cur.fetchall():
        out.append({
            "fonte_aluno":               row["fonte_aluno"],
            "matricula":                 row["matricula"],
            "contract_id":               row["contract_id"],
            "nome_contratante":          row["nome_contratante"],
            "localidade":                row["localidade"],
            "course_name":               row["course_name"],
            "atividade_principal":       row["atividade_principal"],
            "atividades_secundarias":    row["atividades_secundarias"] or ""
        })

    conn.close()
    return out

def store_embeddings_singles_job(
    records: List[Dict[str, Any]],
    embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL
) -> None:
    """
    records: output of sanitize_input_jobs() with keys:
       'fonte_aluno', 'matricula', 'contract_id',
       'nome_contratante','localidade','course_name',
       'atividade_principal','atividades_secundarias'
    embed_func: function(texts: List[str]) -> List[List[float]]
                If None, embeddings are skipped.
    pg_conn_params: DSN string or dict
    """
    # connect to PostgreSQL
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()

    iterator = tqdm(records, desc="Upserting Jobs", unit="job")

    for rec in iterator:
        iterator.set_postfix(
            fonte_aluno=rec.get("fonte_aluno"),
            matricula=rec.get("matricula"),
            contract_id=rec.get("contract_id")
        )

        # build raw input string
        raw = (
            f"{rec['nome_contratante']}\n"
            f"{rec['localidade']}\n"
            f"{rec['course_name']}\n"
            f"{rec['atividade_principal']}\n"
            f"{rec['atividades_secundarias']}"
        ).strip()

        # generate embedding if function provided
        emb = embed_func([raw])[0] if embed_func else None

        # upsert into job_embeddings
        cur.execute(
            """
            INSERT INTO job_embeddings
              (fonte_aluno, matricula, contract_id, raw_input, embedding, last_update)
            VALUES (%s, %s, %s, %s, %s, localtimestamp)
            ON CONFLICT (fonte_aluno, matricula, contract_id) DO UPDATE
              SET raw_input = EXCLUDED.raw_input,
                  embedding  = COALESCE(EXCLUDED.embedding, job_embeddings.embedding),
                  last_update = localtimestamp
            """,
            (
                rec['fonte_aluno'],
                rec['matricula'],
                rec['contract_id'],
                raw,
                emb
            )
        )
        conn.commit()

    cur.close()
    conn.close()

def fetch_embeddings_job(
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
    embedding_column: str = JOBS_EMBEDDING_COLUMN_NAME
) -> np.ndarray:
    """
    Fetches all embeddings from job_embeddings, parsing each stored JSON/vector
    into a NumPy array of shape (N, ORIGINAL_DIM).
    """
    # 1) connect to Postgres
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()
    cur.execute(f"SELECT {embedding_column} FROM job_embeddings")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 2) parse each JSON (or Python literal) into a Python list of floats
    embeddings = []
    for (vec_str,) in rows:
        if isinstance(vec_str, str):
            try:
                vec = json.loads(vec_str)
            except json.JSONDecodeError:
                vec = ast.literal_eval(vec_str)
        else:
            # if stored as a native list or array type
            vec = vec_str
        embeddings.append(vec)

    # 3) stack into a NumPy array
    return np.array(embeddings, dtype=np.float32)


def fetch_single_embedding_job(
    fonte_aluno: str,
    matricula: str,
    contract_id: int,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL
) -> Optional[list[float]]:
    """
    Returns the stored embedding for one job (fonte_aluno, matricula, contract_id)
    as a Python list of floats, or None if not found.
    """
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()
    cur.execute(
        f"SELECT {JOBS_EMBEDDING_COLUMN_NAME} "
        "FROM job_embeddings "
        "WHERE fonte_aluno = %s AND matricula = %s AND contract_id = %s",
        (fonte_aluno, matricula, contract_id)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None

    vec_str = row[0]
    if isinstance(vec_str, str):
        try:
            return json.loads(vec_str)
        except json.JSONDecodeError:
            return ast.literal_eval(vec_str)
    else:
        # if stored as a native Postgres array/vector
        return list(vec_str)

def fetch_embeddings_job_with_metadata(
    embedding_column: str = JOBS_EMBEDDING_COLUMN_NAME,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
    sqlite_db_path: str = SQLITE_URL
) -> pd.DataFrame:
    """
    Fetch fonte_aluno, matricula, contract_id, nome_contratante, localidade, course_name, atividade_principal
    plus embeddings, latent_code, and raw_input, returning a single DataFrame.
    """
    # 1) Metadata from SQLite
    conn_sql = sqlite3.connect(sqlite_db_path)
    conn_sql.row_factory = sqlite3.Row
    cur_sql = conn_sql.cursor()
    cur_sql.execute("""
    SELECT
      i.fonte_aluno,
      i.matricula,
      i.contract_id,
      COALESCE(c.nome_fantasia, c.nome_contratante) AS nome_contratante,
      c.uf || ' ' || c.municipio || ' ' || c.bairro AS localidade,
      s.course_name,
      SUBSTR(c.atividade_principal, 1, INSTR(c.atividade_principal, ' ')-1) AS atividade_principal_code,
      SUBSTR(c.atividade_principal, INSTR(c.atividade_principal, ' ')+1) AS atividade_principal
    FROM internships i
    JOIN companies c ON c.contratante_id = i.contratante_id
    JOIN students s ON s.fonte_aluno = i.fonte_aluno AND s.matricula = i.matricula
    WHERE c.atividade_principal IS NOT NULL
      AND s.course_name <> 'DESCONHECIDO'
    """)
    rows_sql = cur_sql.fetchall()
    cur_sql.close()
    conn_sql.close()
    meta_df = pd.DataFrame([dict(r) for r in rows_sql])

    # 2) Embeddings from Postgres
    conn_pg = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) else psycopg2.connect(**pg_conn_params)
    cur_pg = conn_pg.cursor()
    cur_pg.execute(f"""
    SELECT fonte_aluno, matricula, contract_id, {embedding_column}, latent_code, raw_input, shared_latent_code
      FROM job_embeddings
    """)
    rows_pg = cur_pg.fetchall()
    cur_pg.close()
    conn_pg.close()
    emb_df = pd.DataFrame(rows_pg, columns=[
        'fonte_aluno', 'matricula', 'contract_id',
        embedding_column, 'latent_code', 'raw_input', 'shared_latent_code'
    ])

    # parse JSON/literal embeddings
    def parse_vec(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return ast.literal_eval(v)
        return v

    emb_df[embedding_column] = emb_df[embedding_column].apply(parse_vec)
    emb_df['latent_code']     = emb_df['latent_code'].apply(parse_vec)
    emb_df['shared_latent_code']     = emb_df['shared_latent_code'].apply(parse_vec)

    # 3) Merge metadata and embeddings
    df = pd.merge(
        meta_df,
        emb_df,
        on=['fonte_aluno', 'matricula', 'contract_id'],
        how='inner',
        validate='one_to_one'
    )

    return df

def fetch_job_cluster_centroids(
    pg_conn_params: Union[str, dict] = POSTGRES_URL,
    cluster_table: str = "job_cluster_centroids"
) -> dict[int, np.ndarray]:
    """
    Returns a mapping from cluster_id → centroid (as an ℓ₂‐normalized np.array).
    """
    # 1) Connect
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()

    # 2) Query centroids
    cur.execute(f"SELECT cluster_id, centroid FROM {cluster_table}")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 3) Parse into dict
    centroids: dict[int, np.ndarray] = {}
    for cid, vec in rows:
        # vec may come back as a Python list, pgvector list, or JSON string
        if isinstance(vec, str):
            vec = json.loads(vec)
        centroids[int(cid)] = np.array(vec, dtype=np.float32)
    return centroids

def fetch_embedding_pairs(
    limit: int = 500,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
) -> pd.DataFrame:
    """
    Pulls up to `limit` job–CV rows from the database, along with their text
    and embedding vectors (which may be stored as JSON strings or native arrays).
    """
    sql = """
    SELECT
      je.fonte_aluno,
      je.matricula,
      je.contract_id,
      je.raw_input    AS job_text,
      ce.llm_parsed_raw_input AS cv_text,
      je.embedding    AS job_embedding,
      ce.embedding    AS cv_embedding
    FROM job_embeddings je
    INNER JOIN cv_embeddings ce
      ON ce.fonte_aluno = je.fonte_aluno
     AND ce.matricula    = je.matricula
    WHERE ce.llm_parsed_raw_input NOT LIKE '%%sem experiências%%'
      AND ce.llm_parsed_raw_input NOT LIKE '%%Currículo indisponível%%'
    LIMIT %s
    """

    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    df = pd.read_sql(sql, conn, params=(limit,))
    conn.close()

    # normalize embeddings from string↔list to List[float]
    def _parse(vec):
        if isinstance(vec, str):
            try:
                return json.loads(vec)
            except json.JSONDecodeError:
                return ast.literal_eval(vec)
        return list(vec)

    df["job_embedding"] = df["job_embedding"].map(_parse)
    df["cv_embedding"]  = df["cv_embedding"].map(_parse)
    return df