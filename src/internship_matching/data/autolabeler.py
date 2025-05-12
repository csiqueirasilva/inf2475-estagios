# data/auto_labeler.py

import ast
import json
import os
import joblib
import nltk
import psycopg2
import numpy as np
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from .db import POSTGRES_URL
from sentence_transformers import SentenceTransformer, models

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
PT_STOPWORDS = stopwords.words("portuguese")

CVS_KEYBERT_CLUSTERNAMES_FILE_PATH = "data/models/keybert_cv_cluster_names.json"
CVS_KEYBERT_KMEANS_FILE_PATH = "data/models/keybert_cv_kmeans.joblib"

CVS_TFIDF_CLUSTERNAMES_FILE_PATH = "data/models/tfidf_cv_cluster_names.json"
CVS_TFIDF_KMEANS_FILE_PATH = "data/models/tfidf_cv_kmeans.joblib"

CVS_CTFIDF_CLUSTERNAMES_FILE_PATH = "data/models/ctfidf_cv_cluster_names.json"
CVS_CTFIDF_KMEANS_FILE_PATH = "data/models/ctfidf_cv_kmeans.joblib"

class CVAutoLabeler:
    def __init__(
        self,
        pg_conn_params: Union[str, dict] = POSTGRES_URL,
        n_clusters: int = 100,
    ):
        self.pg = pg_conn_params
        self.n_clusters = n_clusters

        # These get set by fit_kmeans() and name_clusters_*(…)
        self._kmeans:      KMeans               = None
        self._cluster_names: Dict[int, str]      = {}

        # Helpers for naming
        self._tfidf = TfidfVectorizer(
            stop_words=PT_STOPWORDS,
            max_features=5_000
        )

        # opcao 1 para keybert, fazer o nosso

        # # 1) Load the base Transformer
        # word_emb = models.Transformer("neuralmind/bert-base-portuguese-cased")

        # # 2) Add a mean-pooling layer
        # pool = models.Pooling(
        #     word_emb.get_word_embedding_dimension(),
        #     pooling_mode_mean_tokens=True
        # )

        # # 3) Assemble the SentenceTransformer
        # sbert_model = SentenceTransformer(modules=[word_emb, pool])

        # self._kw_model = KeyBERT(model=sbert_model)

        # opcao 2 para keybert, user um pronto
        model = SentenceTransformer("alfaneo/bertimbau-base-portuguese-sts")
        self._kw_model = KeyBERT(model=model)

    # ——————————————————————————————————————————————————————————————
    # 1) Fitting & Naming
    # ——————————————————————————————————————————————————————————————
    def _fetch_texts_and_latents(self):
        """
        Returns lists of raw_input texts and a (N x latent_dim) array of floats,
        parsing any string-serialized vectors into real numeric lists.
        """
        conn = psycopg2.connect(self.pg) if isinstance(self.pg, str) \
               else psycopg2.connect(**self.pg)
        cur = conn.cursor()
        cur.execute("SELECT raw_input, latent_code FROM cv_embeddings")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        texts = []
        latents = []
        for raw, lat in rows:
            texts.append(raw)
            # If lat is already a list (via pgvector adapter), great:
            if isinstance(lat, (list, tuple)):
                latents.append(lat)
            else:
                # Otherwise it’s a string: try JSON first, then fallback to Python literal
                try:
                    lat_list = json.loads(lat)
                except (TypeError, json.JSONDecodeError):
                    lat_list = ast.literal_eval(lat)
                latents.append(lat_list)

        # Now latents is a List[List[float]]; stack into an (N, D) array
        return texts, np.vstack(latents)

    def fit_kmeans(self):
        """Fit KMeans on the latent_code column."""
        _, Z = self._fetch_texts_and_latents()
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42
        )
        self._kmeans.fit(Z)

    def name_clusters_ctfidf(self, top_n: int = 3):
        texts, _ = self._fetch_texts_and_latents()
        labels = self._kmeans.labels_

        # build cluster_docs
        docs = []
        for cid in range(self.n_clusters):
            members = [txt for txt, lbl in zip(texts, labels) if lbl == cid]
            docs.append(" ".join(members))

        # c-TF-IDF vectorization
        vec = TfidfVectorizer(stop_words=PT_STOPWORDS, max_df=0.9, min_df=1, max_features=1000)
        X = vec.fit_transform(docs)
        terms = vec.get_feature_names_out()

        # extract names
        names = {}
        for cid, row in enumerate(X.toarray()):
            top_idxs = row.argsort()[::-1][:top_n]
            names[cid] = " / ".join(terms[i].title() for i in top_idxs)
        self._cluster_names = names

    def name_clusters_tfidf(self, top_n: int = 3):
        """Name each cluster via top TF-IDF terms."""
        if self._kmeans is None:
            raise RuntimeError("Call fit_kmeans() first")
        texts, _ = self._fetch_texts_and_latents()
        labels   = self._kmeans.labels_

        grouped: Dict[int, List[str]] = {i: [] for i in range(self.n_clusters)}
        for txt, cid in zip(texts, labels):
            grouped[cid].append(txt)

        names = {}
        for cid, docs in grouped.items():
            X      = self._tfidf.fit_transform(docs)
            scores = np.asarray(X.sum(axis=0)).ravel()
            terms  = self._tfidf.get_feature_names_out()
            top_terms = [terms[i] for i in np.argsort(scores)[::-1][:top_n]]
            names[cid] = " / ".join(w.title() for w in top_terms)
        self._cluster_names = names

        for cid in range(self.n_clusters):
            self._cluster_names.setdefault(cid, f"Cluster {cid}")

    def name_clusters_keybert(self, top_n: int = 3):
        """Name each cluster via KeyBERT keyphrases."""
        if self._kmeans is None:
            raise RuntimeError("Call fit_kmeans() first")
        texts, _ = self._fetch_texts_and_latents()
        labels   = self._kmeans.labels_

        grouped = {i: [] for i in range(self.n_clusters)}
        for txt, cid in zip(texts, labels):
            grouped[cid].append(txt)

        names = {}
        for cid, docs in grouped.items():
            mega = " ".join(docs)[:10_000]
            kws  = self._kw_model.extract_keywords(
                mega,
                keyphrase_ngram_range=(1,2),
                stop_words=PT_STOPWORDS,
                top_n=top_n
            )
            names[cid] = " / ".join(k for k,_ in kws).title()
        self._cluster_names = names

        for cid in range(self.n_clusters):
            self._cluster_names.setdefault(cid, f"Cluster {cid}")

    # ——————————————————————————————————————————————————————————————
    # 2) Persistence
    # ——————————————————————————————————————————————————————————————
    def save(self, model_path: str, names_path: str):
        """
        Persist KMeans model and cluster names to disk.
        """
        if self._kmeans is None or not self._cluster_names:
            raise RuntimeError("Nothing to save—run fit and name first")

        # Ensure the directories exist
        model_dir = os.path.dirname(model_path)
        names_dir = os.path.dirname(names_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        if names_dir:
            os.makedirs(names_dir, exist_ok=True)

        # 1) Save the sklearn model
        joblib.dump(self._kmeans, model_path)

        # 2) Save the cluster_names dict
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(self._cluster_names, f, ensure_ascii=False, indent=2)

    def save_keybert(self):
        return self.save(CVS_KEYBERT_KMEANS_FILE_PATH, CVS_KEYBERT_CLUSTERNAMES_FILE_PATH)

    def save_tfidf(self):
        return self.save(CVS_TFIDF_KMEANS_FILE_PATH, CVS_TFIDF_CLUSTERNAMES_FILE_PATH)

    def save_ctfidf(self):
        return self.save(CVS_CTFIDF_KMEANS_FILE_PATH, CVS_CTFIDF_CLUSTERNAMES_FILE_PATH)

    @classmethod
    def load(
        cls,
        model_path: str,
        names_path: str,
        pg_conn_params: Union[str, dict] = POSTGRES_URL
    ) -> "CVAutoLabeler":
        """
        Instantiate a labeler from saved model + names.
        KMeans and cluster_names are loaded, so get_auto_label()
        works immediately.
        """
        self = cls(pg_conn_params)
        # 1) Load the KMeans
        self._kmeans = joblib.load(model_path)

        # 2) Load the cluster names
        with open(names_path, "r", encoding="utf-8") as f:
            self._cluster_names = json.load(f)
        return self

    @classmethod
    def load_keybert(cls):
        return cls.load(CVS_KEYBERT_KMEANS_FILE_PATH, CVS_KEYBERT_CLUSTERNAMES_FILE_PATH)

    @classmethod
    def load_tfidf(cls):
        return cls.load(CVS_TFIDF_KMEANS_FILE_PATH, CVS_TFIDF_CLUSTERNAMES_FILE_PATH)

    @classmethod
    def load_ctfidf(cls):
        return cls.load(CVS_CTFIDF_KMEANS_FILE_PATH, CVS_CTFIDF_CLUSTERNAMES_FILE_PATH)

    # ——————————————————————————————————————————————————————————————
    # 3) Prediction
    # ——————————————————————————————————————————————————————————————
    def get_auto_label(self, fonte_aluno: str, matricula: str) -> str:
        """
        Fetches a single CV's latent_code (parsing if needed),
        predicts its cluster, and returns the pre-computed name.
        """
        if self._kmeans is None or not self._cluster_names:
            raise RuntimeError("Model not loaded or clusters not named")

        # 1) Fetch latent_code from DB
        conn = psycopg2.connect(self.pg) if isinstance(self.pg, str) \
               else psycopg2.connect(**self.pg)
        cur = conn.cursor()
        cur.execute(
            "SELECT latent_code FROM cv_embeddings WHERE fonte_aluno=%s AND matricula=%s",
            (fonte_aluno, matricula)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            raise KeyError(f"No latent found for {fonte_aluno}/{matricula}")

        raw = row[0]
        if isinstance(raw, str):
            try:    z_list = json.loads(raw)
            except: z_list = ast.literal_eval(raw)
        else:
            z_list = raw

        z = np.array(z_list, dtype=float).reshape(1, -1)
        cid = int(self._kmeans.predict(z)[0])

        # return name or default
        return self._cluster_names.get(cid, f"Cluster {cid}")