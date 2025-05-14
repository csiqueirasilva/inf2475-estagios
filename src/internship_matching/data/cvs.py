import json
import sqlite3
from typing import Any, Dict, List, Union

import numpy as np
from tqdm import tqdm

from ..data.embed import OLLAMA_LLM

from ..data.db import POSTGRES_URL, SQLITE_URL
from ..utils import get_device
import psycopg2
from typing import Callable, Optional, List, Dict, Union, Any

import time

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

        has_cv_data = True if birthdate == not_available and gender == not_available and interests == not_available and educations == not_available and experiences == not_available and languages == not_available else False

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
                "SELECT 1 FROM cv_embeddings WHERE llm_parsed_raw_input IS NOT NULL AND last_update > now() - interval '12 hour' AND matricula = %s AND fonte_aluno = %s",
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
                        "O currículo a seguir deve ser preparado para futura clusterização com seu texto totalmente em português brasileiro. Sua saída deve ser apenas o texto do currículo formatado em 500 tokens ou menos. Não responda nenhum comentário sobre como o texto é ou foi otimizado, apenas retorne o currículo pronto para clusterização. Se uma informação não estiver disponível, não a inclua. Não comece a saída usando a chave Currículo:. Caso o currículo não disponha de informaçãoes suficientes, apenas use a saída: Currículo indisponível"),
                        ("human", text)
                    ]
                    # single invoke
                    try:
                        parsed = OLLAMA_LLM.invoke(prompt)
                        parsed = OLLAMA_LLM.invoke([
                            ("system", "Reescreva exatamente a entrada, mas traduzindo para português brasileiro."),
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

                    if all_text != '': 

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

                # upsert into PostgreSQL
                cur.execute(
                    """
                    INSERT INTO cv_embeddings
                    (fonte_aluno, matricula, raw_input, llm_parsed_raw_input, embedding, last_update)
                    VALUES (%s, %s, %s, %s, %s, localtimestamp)
                    ON CONFLICT (fonte_aluno, matricula) DO UPDATE
                    SET last_update = localtimestamp,
                        llm_parsed_raw_input = EXCLUDED.llm_parsed_raw_input,
                        embedding = COALESCE(EXCLUDED.embedding, cv_embeddings.embedding)
                    """,
                    (
                        rec["fonte_aluno"],
                        rec["matricula"],
                        rec.get("text", ''),
                        parsed,
                        vec
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