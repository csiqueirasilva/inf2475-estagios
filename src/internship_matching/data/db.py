import sqlite3
from pathlib import Path
import pandas as pd

POSTGRES_URL = "postgresql://postgres:postgres@localhost:5432/ai_test"
SQLITE_URL = "data/processed/data.db"

def init_db(db_path: str, schema_path: str, csv_root_path: str):
    """
    Delete any existing DB file, ensure the parent directory exists,
    create a fresh one, and run the schema.
    """
    db_file = Path(db_path)

    # 1) Make sure the parent folder exists
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # 2) Remove old DB (if any)
    if db_file.exists():
        db_file.unlink()

    # 3) Connect (this will create the file)
    conn = sqlite3.connect(str(db_file))

    # 4) Load and execute your schema.sql
    schema_sql = Path(schema_path).read_text(encoding='utf-8')
    conn.executescript(schema_sql)

    cursor = conn.cursor()

    # Ensure foreign keys are enabled
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Load CSVs
    df1 = pd.read_csv(f"{csv_root_path}/dados-sge-compl-cursos.csv", sep=";", low_memory=False)  # Contains 'id', 'nome'
    df2 = pd.read_csv(f"{csv_root_path}/dados-sge.csv", sep=";", low_memory=False, dtype={"matricula": "string"})  # Contains internship and company data
    df3 = pd.read_csv(f"{csv_root_path}/dados-vagasonline.csv", sep=";", low_memory=False, dtype={"matricula": "string"})  # Contains log/actions and offers

    df1 = trim_all_strings(df1)
    df2 = trim_all_strings(df2)
    df3 = trim_all_strings(df3)

    # === Insert Companies ===
    df1.to_sql("raw_courses", conn, if_exists="replace", index=False)
    df2.to_sql("raw_internships", conn, if_exists="replace", index=False)
    df3.to_sql("raw_offers_and_logs", conn, if_exists="replace", index=False)

    df_sgu_rel = pd.read_csv(
        f"{csv_root_path}/dados-sgu-rel-curso.csv",
        sep=";",
        dtype={"MATRICULA": "string"},
        skipinitialspace=True
    )
    df_sgu_rel.columns = df_sgu_rel.columns.str.lower()
    df_sgu_rel.rename(columns={"matricula": "matricula"}, inplace=True)

    df_sgu_rel = trim_all_strings(df_sgu_rel)

    df_sgu_rel.to_sql("raw_student_course_codes", conn, if_exists="replace", index=False)

    df_sge_rel = pd.read_csv(
        f"{csv_root_path}/dados-sge-rel-curso.csv",
        sep=";",
        dtype=str
    )
    df_sge_rel.columns = df_sge_rel.columns.str.lower()

    df_sge_rel = trim_all_strings(df_sge_rel)

    df_sge_rel.to_sql("raw_course_code_map", conn, if_exists="replace", index=False)

    insert_students_in_chunks(conn)
    insert_companies_in_chunks(conn)
    # some companies did not hired people and had to be fetched separately from the database, into another csv file
    csv_remaining = Path(csv_root_path) / "dados-sge-remaining-companies.csv"
    if csv_remaining.exists():
        insert_remaining_companies_from_csv(conn, str(csv_remaining))
    insert_internships_in_chunks(conn)
    upsert_students_from_cv_csv(conn, f"{csv_root_path}/dados-vagasonline-cvs.csv")
    update_students_course_names(conn)
    insert_job_offers_in_chunks(conn)
    insert_actions_from_logs(conn)

    conn.commit()
    conn.close()

def insert_actions_from_logs(conn, batch_size=1000):
    """
    Extracts student actions from raw_offers_and_logs and inserts them into the actions table.
    Only includes actions where tipo_usuario_fonte = 'ALUNO'.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    query = """
    INSERT OR IGNORE INTO actions (
        log_action,
        dt_log,
        tipo_usuario,
        matricula,
        job_id,
        offer_type,
        fonte
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    sql_source = """
    SELECT DISTINCT
        logs.log_action,
        logs.dt_log_action,
        logs.tipo_usuario_alvo,
        logs.fonte,
        logs.matricula,
        c.nome AS nome_curso,
        logs.offer AS id_job
    FROM raw_offers_and_logs logs
    LEFT JOIN raw_courses c ON c.id = logs.idcurso
    WHERE logs.tipo_usuario_fonte = 'ALUNO'
    """

    for chunk in pd.read_sql(sql_source, conn, chunksize=batch_size):
        str_cols = chunk.select_dtypes(include=["object", "string"]).columns
        chunk[str_cols] = chunk[str_cols].apply(lambda col: col.str.strip())
        chunk = chunk.apply(lambda col: col.map(clean_cell))
        chunk["dt_log_action"] = chunk["dt_log_action"].apply(safe_parse_datetime)

        rows = chunk[[
            "log_action",
            "dt_log_action",
            "tipo_usuario_alvo",
            "matricula",
            "id_job",
            "nome_curso",
            "fonte"
        ]].values.tolist()
        cursor.executemany(query, rows)
        conn.commit()

    cursor.execute("PRAGMA foreign_keys = ON;")

def clean_cell(x):
    if pd.isna(x) or str(x).strip().lower() == "nan":
        return None
    return x

def insert_remaining_companies_from_csv(conn, csv_path: str, batch_size: int = 1000):
    """
    Inserts additional companies from a CSV export into the companies table.
    """
    df = pd.read_csv(csv_path, sep=";", dtype=str)

    # Strip whitespace from strings
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Convert types
    df["id_contratante"] = pd.to_numeric(df["id_contratante"], errors="coerce")
    df["eh_pf"] = df["eh_pf"].map({"true": True, "false": False})
    df["cnpj_invalido"] = df["cnpj_invalido"].map({"true": True, "false": False})
    df["capital_social_receita"] = pd.to_numeric(df["capital_social_receita"], errors="coerce")

    df = df.apply(lambda col: col.map(clean_cell))

    insert_query = """
    INSERT OR IGNORE INTO companies (
        contratante_id,
        nome_contratante,
        eh_pf,
        cnpj_invalido,
        municipio,
        uf,
        bairro,
        nome_fantasia,
        nome_receita,
        abertura,
        situacao,
        tipo,
        dt_atualizacao,
        porte,
        capital_social,
        atividade_principal,
        atividades_secundarias
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        rows = chunk[
            [
                "id_contratante", "nome_contratante", "eh_pf", "cnpj_invalido",
                "municipio_receita", "uf_receita", "bairro_receita",
                "nome_fantasia_receita", "nome_receita", "abertura_receita",
                "situacao_receita", "tipo_receita", "dt_atualizacao_receita",
                "porte_receita", "capital_social_receita",
                "contratante_atividade_principal", "contratante_atividades_secundarias"
            ]
        ].values.tolist()
        cursor.executemany(insert_query, rows)
        conn.commit()

    cursor.execute("PRAGMA foreign_keys = ON;")

def insert_companies_in_chunks(conn, batch_size=1000):
    """
    Inserts companies from raw_internships into the normalized companies table.
    Groups by id_contratante using SELECT DISTINCT.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    query = """
    INSERT OR IGNORE INTO companies (
        contratante_id,
        nome_contratante,
        eh_pf,
        cnpj_invalido,
        municipio,
        uf,
        bairro,
        nome_fantasia,
        nome_receita,
        abertura,
        situacao,
        tipo,
        dt_atualizacao,
        porte,
        capital_social,
        atividade_principal,
        atividades_secundarias
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    sql_source = """
    SELECT DISTINCT
        id_contratante,
        nome_contratante,
        eh_pf,
        cnpj_invalido,
        municipio_receita,
        uf_receita,
        bairro_receita,
        nome_fantasia_receita,
        nome_receita,
        abertura_receita,
        situacao_receita,
        tipo_receita,
        dt_atualizacao_receita,
        porte_receita,
        CAST(capital_social_receita AS REAL) AS capital_social,
        contratante_atividade_principal,
        contratante_atividades_secundarias
    FROM raw_internships
    WHERE id_contratante IS NOT NULL
    """

    for chunk in pd.read_sql(sql_source, conn, chunksize=batch_size):
        str_cols = chunk.select_dtypes(include=["object", "string"]).columns
        chunk[str_cols] = chunk[str_cols].apply(lambda col: col.str.strip())
        chunk = chunk.apply(lambda col: col.map(clean_cell))
        
        rows = chunk.values.tolist()
        cursor.executemany(query, rows)
        conn.commit()

    cursor.execute("PRAGMA foreign_keys = ON;")

def update_students_course_names(conn, batch_size=1000):
    """
    Updates students.course_name for those marked as 'DESCONHECIDO' and fonte_aluno = 'VRADM',
    using raw_student_course_codes and raw_course_code_map to map the actual course names.
    """
    cursor = conn.cursor()

    # Query to get the correct course name mappings
    mapping_query = """
    SELECT s.fonte_aluno, s.matricula, mapc.nome AS course_name
    FROM students s
    INNER JOIN raw_student_course_codes maps ON maps.matricula = s.matricula
    INNER JOIN raw_course_code_map mapc
        ON mapc.sigla_curso = maps.sigla_curso
       AND mapc.sigla_hab = maps.sigla_hab
       AND mapc.sigla_enf = maps.sigla_enf
    WHERE (s.course_name = 'DESCONHECIDO' or s.course_name is null) AND s.fonte_aluno = 'VRADM'
    """

    # Read the mapping data in chunks and update students accordingly
    for chunk in pd.read_sql(mapping_query, conn, chunksize=batch_size):
        rows = chunk.dropna().values.tolist()
        cursor.executemany(
            """
            UPDATE students
            SET course_name = ?
            WHERE fonte_aluno = ? AND matricula = ?
            """,
            [(row[2], row[0], row[1]) for row in rows]  # (course_name, fonte_aluno, matricula)
        )
        conn.commit()

def trim_all_strings(df):
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    return df

def insert_students_in_chunks(conn, batch_size=1000):
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    query = """
    INSERT OR IGNORE INTO students (
        fonte_aluno,
        matricula,
        course_name
    ) VALUES (?, ?, ?)
    """
    for chunk in pd.read_sql(
        "SELECT fonte_aluno, matricula, curso_nome FROM raw_internships "
        "WHERE fonte_aluno IS NOT NULL AND matricula IS NOT NULL",
        conn,
        chunksize=batch_size
    ):
        rows = chunk.drop_duplicates().values.tolist()
        cursor.executemany(query, rows)
        conn.commit()

    cursor.execute("PRAGMA foreign_keys = ON;")

def safe_parse_date(value):
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date().isoformat()
    except Exception:
        return None

def safe_parse_datetime(value):
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def insert_internships_in_chunks(conn, batch_size=1000):
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    query = """
    INSERT OR IGNORE INTO internships (
        contract_id,
        fonte_aluno,
        matricula,
        contratante_id,
        dt_cadastro,
        dt_rescisao,
        remuneracao
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    for chunk in pd.read_sql(
        """
        SELECT contract_id, fonte_aluno, matricula, id_contratante, dt_cadastro, data_rescisao, remuneracao
        FROM raw_internships
        WHERE contract_id IS NOT NULL
        """,
        conn,
        chunksize=batch_size
    ):
        # Convert to datetime safely and format to string
        chunk["dt_cadastro"] = chunk["dt_cadastro"].apply(safe_parse_datetime)
        chunk["data_rescisao"] = chunk["data_rescisao"].apply(safe_parse_date)

        # Replace NaNs or NaT-formatted strings with None
        chunk = chunk.where(pd.notnull(chunk), None)

        # Prepare rows for executemany
        rows = chunk.drop_duplicates(subset="contract_id").values.tolist()
        cursor.executemany(query, rows)
        conn.commit()

    cursor.execute("PRAGMA foreign_keys = ON;")

def insert_job_offers_in_chunks(conn, batch_size=1000):
    """
    Extracts distinct job offers from raw_offers_and_logs and inserts them into job_offers table.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    query = """
    INSERT OR IGNORE INTO job_offers (
        job_id,
        contratante_id,
        date_time,
        requirements,
        activities,
        period_beg,
        period_end,
        city,
        neighbourhood,
        course_names
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    sql_source = """
    SELECT DISTINCT
        id AS job_id,
        sge AS contratante_id,
        date_time,
        requirements,
        activities,
        periodo_beg,
        periodo_end,
        city,
        neighbourhood,
        cursos_da_oferta AS course_names
    FROM raw_offers_and_logs
    WHERE id IS NOT NULL
    """

    for chunk in pd.read_sql(sql_source, conn, chunksize=batch_size):
        # Clean up formatting: remove whitespace and replace NaNs with None
        str_cols = chunk.select_dtypes(include=["object", "string"]).columns
        chunk[str_cols] = chunk[str_cols].apply(lambda col: col.str.strip())

        def clean_cell(x):
            if pd.isna(x) or str(x).strip().lower() == "nan":
                return None
            return x

        chunk = chunk.apply(lambda col: col.map(clean_cell))

        # Convert dates
        chunk["date_time"] = chunk["date_time"].apply(safe_parse_datetime)

        rows = chunk.values.tolist()
        cursor.executemany(query, rows)
        conn.commit()

    cursor.execute("PRAGMA foreign_keys = ON;")

def upsert_students_from_cv_csv(conn, csv_path: str, batch_size: int = 1000):
    import pandas as pd

    df = pd.read_csv(csv_path, sep=";", dtype=str)
    df.columns = df.columns.str.lower().str.strip()

    # Normalize values
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    df = df.apply(lambda col: col.map(lambda x: None if pd.isna(x) or str(x).strip().lower() == "nan" else x))

    # Prepare cursor and SQL
    cursor = conn.cursor()

    update_query = """
    INSERT INTO students (
        fonte_aluno, matricula, name, birthdate, gender, cpf, email,
        tel, cel, address, city, state,
        areas_de_interesse, educations, experiences, languages, course_name
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DESCONHECIDO')
    ON CONFLICT(fonte_aluno, matricula)
    DO UPDATE SET
        name = excluded.name,
        birthdate = excluded.birthdate,
        gender = excluded.gender,
        cpf = excluded.cpf,
        email = excluded.email,
        tel = excluded.tel,
        cel = excluded.cel,
        address = excluded.address,
        city = excluded.city,
        state = excluded.state,
        areas_de_interesse = excluded.areas_de_interesse,
        educations = excluded.educations,
        experiences = excluded.experiences,
        languages = excluded.languages
    """

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        rows = chunk[[
            "fonte", "matricula", "name", "birthdate", "gender", "cpf", "email",
            "tel", "cel", "address", "city", "state",
            "areas_de_interesse", "educations", "experiences", "languages"
        ]].values.tolist()
        cursor.executemany(update_query, rows)
        conn.commit()

def start_database_import(database_path = SQLITE_URL, csv_root_path = "data"):
    db_path = Path(database_path)
    schema_path = Path("sql/schema.sql")
    init_db(str(db_path), str(schema_path), str(csv_root_path))