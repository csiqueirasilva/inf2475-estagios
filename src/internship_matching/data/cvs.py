import json
import sqlite3
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..constants import CVS_EMBEDDING_COLUMN_NAME, CVS_LATENT_SIZE

from ..data.embed import OLLAMA_LLM, get_embed_func

from ..data.db import POSTGRES_URL, SQLITE_URL, fetch_courses_per_student
from ..utils import get_device
import psycopg2
from typing import Callable, Optional, List, Dict, Union, Any

import time

def cv_fill_raw_embeddings(
    pg_conn_params: Union[str, dict] = POSTGRES_URL,
    batch_size: int = 100
) -> None:
    """
    Fetch rows whose raw_embedding IS NULL, compute new embeddings with get_embed_func(),
    and write them back into cv_embeddings.raw_embedding using (fonte_aluno, matricula) as key.
    The embed_func takes a list of raw_input strings and returns a list of embedding vectors.
    Shows progress with tqdm.
    """
    embed_func = get_embed_func()

    # Establish connection
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)

    try:
        with conn:
            with conn.cursor() as cur:
                # Get total count for progress bar
                cur.execute("""
                    SELECT COUNT(*)
                    FROM cv_embeddings
                    WHERE raw_embedding IS NULL
                """)
                total_rows = cur.fetchone()[0]

                select_sql = """
                    SELECT fonte_aluno, matricula, raw_input
                    FROM cv_embeddings
                    WHERE raw_embedding IS NULL
                    ORDER BY fonte_aluno, matricula
                    LIMIT %s
                """
                update_sql = """
                    UPDATE cv_embeddings
                    SET raw_embedding = %s
                    WHERE fonte_aluno = %s
                      AND matricula    = %s
                """

                processed = 0
                pbar = tqdm(total=total_rows, desc="Embedding CVs", unit="row")

                while True:
                    cur.execute(select_sql, (batch_size,))
                    rows = cur.fetchall()
                    if not rows:
                        break

                    # Prepare batch for embedding
                    keys = [(fa, mat) for fa, mat, _ in rows]
                    texts = [raw for _, _, raw in rows]

                    # Compute embeddings for the batch
                    embeddings = embed_func(texts)  # List[List[float]]

                    # Update each row with its embedding
                    for (fa, mat), vec in zip(keys, embeddings):
                        vec_list = [float(x) for x in vec]
                        cur.execute(update_sql, (json.dumps(vec_list), fa, mat))
                    
                    conn.commit()

                    processed_batch = len(rows)
                    processed += processed_batch
                    pbar.update(processed_batch)

                pbar.close()
                print(f"✅ Populated for cv raw_embeddings for {processed} rows.")
    finally:
        conn.close()

def update_raw_embeddings(): 
    """
    Updates the raw embeddings in the cv_embeddings table.
    This is a temporary function to be removed in the future.
    """
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    cur.execute("""
        UPDATE cv_embeddings
        SET raw_embedding = embedding
        WHERE raw_embedding IS NULL
    """)
    conn.commit()
    cur.close()
    conn.close()

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

    not_available = "[NO AVAILABLE DATA]"

    def process_multivalue(value: str) -> List[str]:
        if not value:
            return [not_available]
        items = {item.strip() for item in value.split("|||") if item.strip()}
        return list(items) if items else [not_available]

    out = []
    for row in cur.fetchall():
        # required (never null)
        fonte     = row["fonte_aluno"]
        matricula = row["matricula"]
        course    = row["course_name"]

        # optional single-value
        birthdate = row["birthdate"] or not_available
        gender    = row["gender"]    or not_available

        # multi-value, de-duplicated
        interests   = ", ".join(process_multivalue(row["areas_de_interesse"]))
        educations  = ", ".join(process_multivalue(row["educations"]))
        experiences = ", ".join(process_multivalue(row["experiences"]))
        languages   = ", ".join(process_multivalue(row["languages"]))

        has_cv_data = (
            (birthdate != not_available and birthdate) or 
            (gender != not_available and gender) or 
            (interests != not_available and interests) or
            (educations != not_available and educations) or
            (experiences != not_available and experiences) or
            (languages != not_available and languages)
        )

        text = (
            f"Course: {course}. Birthdate: {birthdate}. Gender: {gender}. "
            f"Interests: {interests}. Educations: {educations}. "
            f"Experiences: {experiences}. Languages: {languages}."
        )

        out.append({
            "fonte_aluno": fonte,
            "matricula": matricula,
            "text": text,
            "course": course,
            "has_cv_data": has_cv_data
        })

    conn.close()
    return out

def store_embeddings_singles_cv(
    records: List[Dict[str, Any]],
    embed_func: Optional[Callable[[str], List[float]]] = None,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
    sqlite_db_path: str = SQLITE_URL
) -> None:
    """
    records: output of sanitize_input_cvs() with keys 'fonte_aluno','matricula','text'
    pg_conn_params: DSN string or dict
    embed_func: function(text:str) -> List[float]
                If None, embeddings are skipped.
    """
    # connect to PostgreSQL
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()

    # connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()

    # iterate one by one, showing fonte_aluno and matricula
    with tqdm(records, desc="Upserting CVs", unit="cv") as pbar:
        for rec in pbar:
            pbar.set_postfix(fonte_aluno=rec["fonte_aluno"], matricula=rec["matricula"])

            cur.execute(
                """
                SELECT 1 
                FROM cv_embeddings 
                WHERE 
                    llm_parsed_raw_input IS NOT NULL AND
                    last_update > now() - interval '24 hours' AND
                    matricula = %s AND fonte_aluno = %s
                """,
                (rec["matricula"], rec["fonte_aluno"])
            )
            exists_recent = cur.fetchone() is not None

            if not exists_recent:

                has_cv_data = rec.get("has_cv_data", False)

                if has_cv_data:
                    text = rec["text"]
                    # prepare prompt for LLM call
                    prompt = [
                        ("system", 
                        "As informações de entrada do usuário vieram de um currículo. Você deve preparar um texto para futura clusterização, e o texto de saída deve ser totalmente em português brasileiro. Sua saída deve ser apenas o texto do formatado em 500 tokens ou menos. Não responda nenhum comentário sobre como o texto é ou foi otimizado, apenas retorne o currículo pronto para clusterização. Se uma informação não estiver disponível, não a inclua. Não comece a saída usando a chave Currículo:. Se o curso for desconhecido, use a saída: Currículo indisponível. Caso apenas o curso seja conhecido e não houver nenhuma outra informação disponível, use a saída: Aluno cursando [curso] sem experiências. Caso o curso seja DESCONHECIDO e nenhuma outra informação esteja disponível, retorna: Currículo indisponível."),
                        ("human", text)
                    ]
                    # single invoke
                    try:
                        parsed = OLLAMA_LLM.invoke(prompt)
                        parsed = OLLAMA_LLM.invoke([
                            ("system", "Reescreva exatamente a entrada, mas traduzindo para português brasileiro. Apenas retorne a tradução."),
                            ("human", parsed)
                        ])
                    except:
                        parsed = "Currículo indisponível"

                else:
                    curso = rec.get("course", "")
                    # retrieve internship-derived CV data from SQLite
                    sqlite_cur.execute("""
    WITH internships_grouped AS (
    SELECT 
        MIN(ri.dt_inicio) AS dt_inicio,
        MAX(COALESCE(i.dt_rescisao, ri.dt_termino)) AS dt_termino,
        CASE WHEN ri.vale_transporte = 0 THEN 'não possui vale transporte' ELSE CAST(ri.valor_vale_transporte AS TEXT) END AS vl_vale_transporte,
        ri.horas_semanais,
        COALESCE(ri.remuneracao, 'remuneração não informada') AS remuneracao,
        ri.nome_contratante AS razao_social,
        ri.nome_receita AS nome_fantasia,
        ri.abertura_receita AS dt_abertura_empresa,
        ri.municipio_receita,
        ri.uf_receita,
        ri.bairro_receita,
        ri.situacao_receita AS situacao_cadastro_receita_federal_contratante,
        ri.tipo_receita AS tipo_empresarial_receita,
        COALESCE(ri.capital_social_receita, 'capital social não informado') AS capital_social_receita,
        ri.contratante_atividade_principal,
        ri.dias_de_estagio
    FROM internships i
    LEFT JOIN companies c ON c.contratante_id = i.contratante_id
    LEFT JOIN raw_internships ri ON ri.contract_id = i.contract_id
    WHERE i.fonte_aluno = ? AND i.matricula = ?
    GROUP BY
        CASE WHEN ri.vale_transporte = 0 THEN 'não possui vale transporte' ELSE CAST(ri.valor_vale_transporte AS TEXT) END,
        ri.horas_semanais,
        COALESCE(ri.remuneracao, 'remuneração não informada'),
        ri.nome_contratante,
        ri.nome_receita,
        ri.abertura_receita,
        ri.municipio_receita,
        ri.uf_receita,
        ri.bairro_receita,
        ri.situacao_receita,
        ri.tipo_receita,
        COALESCE(ri.capital_social_receita, 'capital social não informado'),
        ri.contratante_atividade_principal,
        ri.dias_de_estagio
    ), intern_text AS (
    SELECT
        'dt_inicio: ' || dt_inicio || ', ' ||
        'dt_termino: ' || dt_termino || ', ' ||
        'vl_vale_transporte: ' || vl_vale_transporte || ', ' ||
        'horas_semanais: ' || horas_semanais || ', ' ||
        'remuneracao: ' || remuneracao || ', ' ||
        'razao_social: ' || razao_social || ', ' ||
        'nome_fantasia: ' || nome_fantasia || ', ' ||
        'dt_abertura_empresa: ' || dt_abertura_empresa || ', ' ||
        'municipio_receita: ' || municipio_receita || ', ' ||
        'uf_receita: ' || uf_receita || ', ' ||
        'bairro_receita: ' || bairro_receita || ', ' ||
        'situacao_cadastro_receita_federal_contratante: ' || situacao_cadastro_receita_federal_contratante || ', ' ||
        'tipo_empresarial_receita: ' || tipo_empresarial_receita || ', ' ||
        'capital_social_receita: ' || capital_social_receita || ', ' ||
        'contratante_atividade_principal: ' || contratante_atividade_principal || ', ' ||
        'dias_de_estagio: ' || dias_de_estagio AS col,
        dt_termino
    FROM internships_grouped
    ORDER BY dt_termino ASC
    )
    SELECT group_concat(col, '----------------') AS all_internships_text
    FROM intern_text;
    """, (rec["fonte_aluno"], rec["matricula"]))
                    all_text = sqlite_cur.fetchone()["all_internships_text"] or ''

                    if curso == 'DESCONHECIDO':
                        parsed = "Currículo indisponível"
                    elif all_text != '': 

                        cv_data_sqlite_string = f"Curso: {curso}\nInformações de experiências de estágio: \n{all_text}"

                        # prepare prompt for LLM call
                        prompt = [
                            ("system", 
                            "A partir dos dados a seguir você deve escrever um texto resumido em português brasileiro, como se fosse um relato em busca de uma oportunidade profissional de forma sucinta, mas que deve necessariamene conter atividades realizadas em cada uma das experiências - use as informações de atividade principal e secundárias para redigir que atividades foram desenvolvidas. Use um formato de saída apropriado para futura clusterização. Sua saída deve ser apenas o texto formatado em 500 tokens ou menos. Não responda nenhum comentário sobre como o texto é ou foi otimizado, apenas retorne o currículo pronto para clusterização. Não use qualquer placeholder textual - sua saída não é um template. Se uma informação não estiver disponível, não a inclua. Não comece a saída usando a chave Currículo:. Caso o texto do usuário não disponha de informaçãoes suficientes, apenas use a saída: Currículo indisponível"),
                            ("human", cv_data_sqlite_string)
                        ]
                        try:
                            parsed = OLLAMA_LLM.invoke(prompt)
                            parsed = OLLAMA_LLM.invoke([
                                ("system", "Reescreva exatamente a entrada, mas traduzindo para português brasileiro."),
                                ("human", parsed)
                            ])
                        except:
                            parsed = "Currículo indisponível"
                        
                    else: 
                        parsed = f"Aluno cursando {curso} sem experiências"

                # compute embedding for this single result
                vec = embed_func([parsed])[0] if embed_func else None
                vec2 = embed_func([rec.get("text", '')])[0] if embed_func else None

                # upsert into PostgreSQL
                cur.execute(
                    """
                    INSERT INTO cv_embeddings
                    (fonte_aluno, matricula, raw_input, llm_parsed_raw_input, embedding, raw_embedding, last_update)
                    VALUES (%s, %s, %s, %s, %s, %s, localtimestamp)
                    ON CONFLICT (fonte_aluno, matricula) DO UPDATE
                    SET last_update = localtimestamp,
                        llm_parsed_raw_input = EXCLUDED.llm_parsed_raw_input,
                        embedding = COALESCE(EXCLUDED.embedding, cv_embeddings.embedding),
                        raw_embedding = COALESCE(EXCLUDED.raw_embedding, cv_embeddings.raw_embedding)
                    """,
                    (
                        rec["fonte_aluno"],
                        rec["matricula"],
                        rec.get("text", ''),
                        parsed,
                        vec,
                        vec2
                    )
                )
                conn.commit()

    sqlite_conn.close()
    cur.close()
    conn.close()

def store_embeddings_cv(
    records: List[Dict[str, Any]],
    embed_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
    batch_size: int = 10
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

            # prepare prompts for batch LLM call
            prompts = [
                [
                    ("system", 
                     "O currículo a seguir deve ser preparado para futura clusterização com seu texto totalmente em português brasileiro. Sua saída deve ser apenas o texto do currículo formatado. Caso o currículo não disponha de informaçãoes suficientes, apenas use a saída: Currículo indisponível."),
                    ("human", text)
                ]
                for text in texts
            ]

            # batch invoke: parallelizes individual invokes under the hood
            responses = OLLAMA_LLM.batch(prompts)

            # get embeddings once per batch (or all None)
            vecs = embed_func(responses) if embed_func else [None] * len(batch)

            # upsert each record
            for rec, parsed, vec in zip(batch, responses, vecs):
                cur.execute(
                    """
                    INSERT INTO cv_embeddings
                      (fonte_aluno, matricula, raw_input, llm_parsed_raw_input, embedding, last_update)
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT (fonte_aluno, matricula) DO UPDATE
                      SET last_update = now(), raw_input             = EXCLUDED.raw_input,
                          llm_parsed_raw_input  = EXCLUDED.llm_parsed_raw_input,
                          embedding             = COALESCE(EXCLUDED.embedding, cv_embeddings.embedding)
                    """,
                    (
                        rec["fonte_aluno"],
                        rec["matricula"],
                        rec["text"],
                        parsed.content if hasattr(parsed, 'content') else parsed,
                        vec
                    )
                )
                pbar.update(1)

            conn.commit()

    cur.close()
    conn.close()

def reset_database_embeddings_size_cv(
    pg_conn_params: Union[str, dict] = POSTGRES_URL,
    latent_size: int = CVS_LATENT_SIZE
) -> None:
    """
    Wipes out the old latent_code column, then alters its type to vector(latent_size).
    """
    # Connect
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    # We need DDL, so either autocommit or explicit commit after each
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # 1) Null-out the old values
            cur.execute("UPDATE cv_embeddings SET latent_code = NULL;")
            # 2) Alter the column type; since everything is NULL, the USING cast is trivial
            cur.execute(
                f"""
                ALTER TABLE cv_embeddings
                ALTER COLUMN latent_code
                TYPE vector({latent_size})
                USING latent_code::vector({latent_size});
                """
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def fetch_experiences_per_student(
    embedding_column: str = CVS_EMBEDDING_COLUMN_NAME,
    pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
) -> pd.DataFrame:
    """
    Fetch fonte_aluno, matricula and whether the student has experience (tem_experiencia)
    from Postgres, based on the llm_parsed_raw_input flag.

    Returns a DataFrame with columns:
      ['fonte_aluno', 'matricula', 'tem_experiencia']
    where tem_experiencia is a boolean.
    """
    # 1) open Postgres connection
    conn = (
        psycopg2.connect(pg_conn_params)
        if isinstance(pg_conn_params, str)
        else psycopg2.connect(**pg_conn_params)
    )

    # 2) define and run the query
    query = f"""
        SELECT
          fonte_aluno,
          matricula,
          NOT (llm_parsed_raw_input LIKE '%Aluno cursando%sem experi_ncias%' or llm_parsed_raw_input like '%Currículo indisponível%')
            AS tem_experiencia,
        {embedding_column}
        FROM cv_embeddings;
    """
    df = pd.read_sql_query(query, conn)
    
    df[embedding_column] = df[embedding_column].apply(json.loads)

    # 3) close connection and return
    conn.close()
    return df

def fetch_embeddings_cv_with_courses_filtered_with_experience(
    embedding_column: str = CVS_EMBEDDING_COLUMN_NAME,
    pg_conn_params: Union[str, dict] = POSTGRES_URL,
) -> pd.DataFrame:
    """
    Fetch cv_embeddings.fonte_aluno, matricula, <embedding_column> from Postgres,
    parse the JSON arrays into Python lists, then inner-join to courses_df
    (which must have fonte_aluno, matricula, course_name).

    Returns a DataFrame with columns:
      ['fonte_aluno', 'matricula', 'course_name', embedding_column]
    where embedding_column holds a Python list (or np.array, if you .apply(np.array)).
    """

    courses_df = fetch_courses_per_student()

    # 1) pull raw embedding strings from Postgres
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT fonte_aluno, matricula, {embedding_column}, latent_code, llm_parsed_raw_input as text
        FROM cv_embeddings
        WHERE llm_parsed_raw_input not LIKE '%Aluno cursando%sem experi_ncias%' and llm_parsed_raw_input not like '%Currículo indisponível%'
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 2) build a DataFrame
    emb_df = pd.DataFrame(rows, columns=["fonte_aluno", "matricula", embedding_column, "latent_code", "text"])

    # 3) parse JSON text → list of floats
    emb_df[embedding_column] = emb_df[embedding_column].apply(json.loads)
    emb_df["latent_code"] = emb_df["latent_code"].apply(json.loads)

    # 4) join to your courses_df
    merged = pd.merge(
        courses_df,
        emb_df,
        on=["fonte_aluno", "matricula"],
        how="inner",    # only keep students with both an embedding and a course
        validate="one_to_one"
    )

    return merged

def fetch_embeddings_cv_with_courses(
    embedding_column: str = CVS_EMBEDDING_COLUMN_NAME,
    pg_conn_params: Union[str, dict] = POSTGRES_URL,
) -> pd.DataFrame:
    """
    Fetch cv_embeddings.fonte_aluno, matricula, <embedding_column> from Postgres,
    parse the JSON arrays into Python lists, then inner-join to courses_df
    (which must have fonte_aluno, matricula, course_name).

    Returns a DataFrame with columns:
      ['fonte_aluno', 'matricula', 'course_name', embedding_column]
    where embedding_column holds a Python list (or np.array, if you .apply(np.array)).
    """

    courses_df = fetch_courses_per_student()

    # 1) pull raw embedding strings from Postgres
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT fonte_aluno, matricula, {embedding_column}, latent_code, llm_parsed_raw_input as text
        FROM cv_embeddings
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 2) build a DataFrame
    emb_df = pd.DataFrame(rows, columns=["fonte_aluno", "matricula", embedding_column, "latent_code", "text"])

    # 3) parse JSON text → list of floats
    emb_df[embedding_column] = emb_df[embedding_column].apply(json.loads)
    emb_df["latent_code"] = emb_df["latent_code"].apply(json.loads)

    # 4) join to your courses_df
    merged = pd.merge(
        courses_df,
        emb_df,
        on=["fonte_aluno", "matricula"],
        how="inner",    # only keep students with both an embedding and a course
        validate="one_to_one"
    )

    return merged

def fetch_embeddings_cv(
    pg_conn_params: Union[str, dict] = POSTGRES_URL,
    embedding_column: str = CVS_EMBEDDING_COLUMN_NAME
) -> np.ndarray:
    """
    Fetches all embeddings from cv_embeddings, parsing the string
    representation into a NumPy array of shape (N, 768).
    """
    # 1) Connect to Postgres
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()
    cur.execute(f"SELECT {embedding_column} FROM cv_embeddings")
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
    
def fetch_single_shared_latent_cv(fonte_aluno: str, matricula: str) -> list[float] | None:
    """
    Grab the pre-computed shared_latent_code (vector(96)) for one CV.
    """
    import psycopg2, json
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT shared_latent_code FROM cv_embeddings "
        "WHERE fonte_aluno = %s AND matricula = %s",
        (fonte_aluno, matricula)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row or row[0] is None:
        return None
    # Depending on your driver, this may already be a Python list;
    # if it's a JSON/text, do json.loads
    return row[0] if isinstance(row[0], list) else json.loads(row[0])

def fetch_single_shared_latent_job(contract_id: int) -> list[float] | None:
    import psycopg2, json
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT shared_latent_code FROM job_embeddings WHERE contract_id = %s",
        (contract_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row or row[0] is None:
        return None
    return row[0] if isinstance(row[0], list) else json.loads(row[0])