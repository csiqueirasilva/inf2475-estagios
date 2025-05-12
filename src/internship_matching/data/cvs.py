import json
import sqlite3
from typing import Any, Dict, List, Union

import numpy as np
from tqdm import tqdm

from ..data.db import POSTGRES_URL, SQLITE_URL
from ..utils import get_device
import psycopg2
from typing import Callable, Optional, List, Dict, Union, Any

def sanitize_input_cvs(sqlite_db_path: str = SQLITE_URL) -> List[Dict]:
    """
    Connect to SQLite, pull every student,
    and build a sanitized 'text' for embedding.
    Returns a list of dicts with keys:
       'fonte_aluno', 'matricula', 'text'
    """
    conn = sqlite3.connect(sqlite_db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
    SELECT fonte_aluno, matricula,
           course_name, birthdate, gender,
           email, tel, cel,
           areas_de_interesse, educations, experiences, languages
      FROM students
    """)

    def process_multivalue(value: str) -> List[str]:
        if not value:
            return ["[NO AVAILABLE DATA]"]
        items = {item.strip() for item in value.split("|||") if item.strip()}
        return list(items) if items else ["[NO AVAILABLE DATA]"]

    out = []
    for row in cur.fetchall():
        # required (never null)
        fonte     = row["fonte_aluno"]
        matricula = row["matricula"]
        course    = row["course_name"]

        # optional single-value
        birthdate = row["birthdate"] or "[NO AVAILABLE DATA]"
        gender    = row["gender"]    or "[NO AVAILABLE DATA]"

        # contact flags
        email_flag = "EMAIL: FILLED" if row["email"] else "EMAIL: EMPTY"
        tel_flag   = "TEL: FILLED"   if row["tel"]   else "TEL: EMPTY"
        cel_flag   = "CEL: FILLED"   if row["cel"]   else "CEL: EMPTY"

        # multi-value, de-duplicated
        interests   = ", ".join(process_multivalue(row["areas_de_interesse"]))
        educations  = ", ".join(process_multivalue(row["educations"]))
        experiences = ", ".join(process_multivalue(row["experiences"]))
        languages   = ", ".join(process_multivalue(row["languages"]))

        text = (
            f"Course: {course}. Birthdate: {birthdate}. Gender: {gender}. "
            f"{email_flag}. {tel_flag}. {cel_flag}. "
            f"Interests: {interests}. Educations: {educations}. "
            f"Experiences: {experiences}. Languages: {languages}."
        )

        out.append({
            "fonte_aluno": fonte,
            "matricula": matricula,
            "text": text
        })

    conn.close()
    return out

def store_embeddings_cv(
    records: List[Dict[str, Any]],
    embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
    batch_size: int = 64
) -> None:
    """
    records: output of sanitize_input_cvs() with keys 'fonte_aluno','matricula','text'
    pg_conn_params: DSN string or dict
    embed_func: function(texts:List[str]) -> List[List[float]]
                If None, embeddings are skipped.
    batch_size:  how many CVs to embed in one call to embed_func
    """
    # connect
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()

    total = len(records)
    with tqdm(total=total, desc="Upserting CVs", unit="cv") as pbar:
        for i in range(0, total, batch_size):
            batch = records[i : i + batch_size]
            texts = [r["text"] for r in batch]

            # get vectors once per batch
            vecs = embed_func(texts) if embed_func else [None]*len(batch)

            for rec, vec in zip(batch, vecs):
                cur.execute(
                    """
                    INSERT INTO cv_embeddings (fonte_aluno, matricula, raw_input, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (fonte_aluno, matricula) DO UPDATE
                      SET raw_input = EXCLUDED.raw_input,
                          embedding  = COALESCE(EXCLUDED.embedding, cv_embeddings.embedding)
                    """,
                    (rec["fonte_aluno"], rec["matricula"], rec["text"], vec)
                )
                pbar.update(1)

            # commit after each batch
            conn.commit()

    cur.close()
    conn.close()

def fetch_embeddings_cv(
    pg_conn_params: Union[str, dict] = POSTGRES_URL
) -> np.ndarray:
    """
    Fetches all embeddings from cv_embeddings, parsing the string
    representation into a NumPy array of shape (N, 768).
    """
    # 1) Connect to Postgres
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM cv_embeddings")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 2) Parse each row’s string into a Python list of floats
    embeddings = []
    for (vec_str,) in rows:
        # JSON loads is preferred for JSON-compatible arrays
        vec = json.loads(vec_str)
        embeddings.append(vec)

    # 3) Stack into a NumPy array
    return np.array(embeddings, dtype=np.float32)

def fetch_single_embedding_cv(
    fonte_aluno: str,
    matricula: str,
    pg_conn_params: Union[str, dict] = POSTGRES_URL
) -> Optional[list[float]]:
    """
    Returns the stored 768-dim embedding for one CV as a list of floats,
    or None if not found.
    """
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()
    cur.execute(
        "SELECT embedding FROM cv_embeddings WHERE fonte_aluno=%s AND matricula=%s",
        (fonte_aluno, matricula)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None

    vec_str = row[0]  # this is a JSON string like "[0.1, -0.2, …]"
    try:
        return json.loads(vec_str)
    except json.JSONDecodeError:
        # fallback if it's Python literal syntax
        import ast
        return ast.literal_eval(vec_str)