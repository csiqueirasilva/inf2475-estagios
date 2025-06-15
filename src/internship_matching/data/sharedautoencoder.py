"""
Unified auto-encoder that learns one latent space for *both* CV and Job
embeddings (all come as 768-d Nomic vectors).

Path constants live in constants.py:
    SHARED_AE_FILE_PATH          – .pt weights
    SHARED_LATENT_SIZE           – default 96
    SHARED_HIDDEN_SIZE           – default 384   (or tweak)

After training you can reuse .encoder everywhere instead of having
separate cv.encoder and job.encoder.
"""

from __future__ import annotations
import json, pathlib, math
from typing import Dict, Any, List, Union, Callable

from psycopg2.extras import execute_batch
import numpy as np
import psycopg2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from ..data.db import POSTGRES_URL

from ..constants import (
    SHARED_AE_FILE_PATH,
    SHARED_HIDDEN_SIZE,
    SHARED_LATENT_SIZE,
    GLOBAL_RANDOM_SEED,
)

from .jobs import TORCH_DEVICE, fetch_embeddings_job
from .cvs  import fetch_embeddings_cv

import ast, json, numpy as np, torch

def _coerce_vec(raw) -> list[float]:
    """Return a plain Python list[float] regardless of the DB cell type."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        return list(raw)
    if isinstance(raw, str):
        try:                       # fast path: valid JSON
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)   # fall-back: Python repr
    raise TypeError(f"Un-supported type: {type(raw)}")

class CVJobSharedAutoencoder(nn.Module):
    """
    One encoder / decoder for *all* 768-d CV and Job vectors.
    """

    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = SHARED_HIDDEN_SIZE,
                 latent_dim: int = SHARED_LATENT_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
        )
        self.to(TORCH_DEVICE)

    # ─────────────────────────── utilities ────────────────────────────
    @staticmethod
    def _auto_loss(x, recon, alpha: float = 1.0) -> tuple[torch.Tensor, Dict[str, float]]:
        """Hybrid MSE + cosine distance (weighted by alpha)."""
        mse  = F.mse_loss(recon, x, reduction="mean")
        cosd = 1 - F.cosine_similarity(recon, x, dim=1).mean()
        return mse + alpha * cosd, {"mse": mse.item(), "cos": cosd.item()}

    # ─────────────────────────── training ─────────────────────────────
    @classmethod
    def train_shared(
        cls,
        epochs: int = 25,
        batch_size: int = 128,
        lr: float = 1e-3,
        hidden_dim: int = SHARED_HIDDEN_SIZE,
        latent_dim: int = SHARED_LATENT_SIZE,
        train_frac: float = 0.9,
        seed: int = GLOBAL_RANDOM_SEED,
        alpha: float = 1.0,          # weight for cosine part of the loss
    ) -> "CVJobSharedAutoencoder":

        torch.manual_seed(seed); np.random.seed(seed)

        # 1) fetch & stack data (CVs + Jobs)  →  (N, 768)
        embeds_cv  = fetch_embeddings_cv()
        embeds_job = fetch_embeddings_job()
        X = np.vstack([embeds_cv, embeds_job]).astype(np.float32)

        # 2) split train / val
        tensor = torch.from_numpy(X)
        ds = TensorDataset(tensor)
        N = len(ds); n_train = int(train_frac * N)
        train_ds, val_ds = random_split(ds, [n_train, N-n_train],
                                        generator=torch.Generator().manual_seed(seed))

        tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        vl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # 3) model + optimiser
        model = cls(input_dim=X.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)

        # 4) training loop
        for epoch in range(1, epochs+1):
            model.train(); tr_mse = tr_cos = 0.
            for (batch,) in tqdm(tl, desc=f"E{epoch}/{epochs}", unit="b"):
                batch = batch.to(TORCH_DEVICE)
                recon, _ = model(batch)
                loss, parts = cls._auto_loss(batch, recon, alpha)
                opt.zero_grad(); loss.backward(); opt.step()
                tr_mse += parts["mse"]*len(batch); tr_cos += parts["cos"]*len(batch)
            tr_mse /= n_train; tr_cos /= n_train

            # val
            model.eval(); val_mse = val_cos = 0.
            with torch.no_grad():
                for (batch,) in vl:
                    batch = batch.to(TORCH_DEVICE)
                    recon, _ = model(batch)
                    l, p = cls._auto_loss(batch, recon, alpha)
                    val_mse += p["mse"]*len(batch); val_cos += p["cos"]*len(batch)
            val_mse /= (N-n_train); val_cos /= (N-n_train)
            print(f"Epoch {epoch:2d} | "
                  f"Train MSE {tr_mse:.4e} Cos {tr_cos:.4e}  | "
                  f"Val MSE {val_mse:.4e} Cos {val_cos:.4e}")

        torch.save(model.state_dict(), SHARED_AE_FILE_PATH)
        print(f"✅ Shared AE saved to {SHARED_AE_FILE_PATH}")
        return model

    # ─────────────────────────── helpers ──────────────────────────────
    @classmethod
    def load(cls,
             path: str = SHARED_AE_FILE_PATH,
             input_dim: int = 768,
             hidden_dim: int = SHARED_HIDDEN_SIZE,
             latent_dim: int = SHARED_LATENT_SIZE,
             device: Union[str, torch.device] = TORCH_DEVICE) -> "CVJobSharedAutoencoder":
        m = cls(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device).eval()
        return m

    def encode(self, vecs: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return latent codes (np.ndarray, shape (N, latent_dim))."""
        if isinstance(vecs, np.ndarray):
            t = torch.tensor(vecs, dtype=torch.float32, device=TORCH_DEVICE)
        else:
            t = vecs.to(TORCH_DEVICE)
        with torch.no_grad():
            z = self.encoder(t)
        return z.cpu().numpy()

    def round_trip(self, vecs: np.ndarray) -> float:
        """Mean reconstruction MSE on supplied 768-d vectors."""
        t = torch.tensor(vecs, dtype=torch.float32, device=TORCH_DEVICE)
        with torch.no_grad():
            recon, _ = self(t)
        return F.mse_loss(recon, t).item()

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    @classmethod
    def generate_all_latents(
        cls,
        pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
        batch_size: int = 512,
        device: torch.device = TORCH_DEVICE,
    ) -> None:
        """
        Compute 96-d shared latent codes for *both* cv_embeddings and
        job_embeddings and upsert them into the `shared_latent_code` column.
        """

        def _process(table: str, pks: list[str], emb_col: str) -> None:
            """Inner helper that works for either table."""
            pk_sql = ", ".join(pks)
            sel_sql = f"""
                SELECT {pk_sql}, {emb_col}
                FROM {table}
                WHERE {emb_col} IS NOT NULL
            """
            upd_sql = f"""
                UPDATE {table}
                SET shared_latent_code = %s
                WHERE {pk_sql.replace(',', ' = %s AND')} = %s
            """

            cur.execute(f"SELECT COUNT(*) FROM ({sel_sql}) AS q")
            total = cur.fetchone()[0]
            offset = 0

            with tqdm(total=total, desc=f"Latents→{table}", unit="rows") as bar:
                while offset < total:
                    cur.execute(f"{sel_sql} ORDER BY {pk_sql} LIMIT %s OFFSET %s",
                                (batch_size, offset))
                    rows = cur.fetchall()
                    if not rows:
                        break

                    ids = [r[:-1] for r in rows]         # all PK columns except the embedding
                    raw_embeds = [r[-1] for r in rows]   # the embedding column
                    embeds     = [_coerce_vec(e) for e in raw_embeds]

                    x = torch.tensor(np.asarray(embeds, dtype=np.float32), device=device)
                    with torch.no_grad():
                        _, z = ae(x)
                    latents = z.cpu().tolist()

                    # build list[tuple] matching the order of placeholders:
                    # (%s)  ← latent
                    # (%s)  ← pk1
                    # (%s)  ← pk2 ...
                    updates = [(lat, *pk) for lat, pk in zip(latents, ids)]
                    execute_batch(cur, upd_sql, updates)
                    conn.commit()
                    offset += len(rows)
                    bar.update(len(rows))

        # ── load AE & connect ──────────────────────────────────────────────────
        ae = cls.load(SHARED_AE_FILE_PATH)
        ae.eval()

        conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
            else psycopg2.connect(**pg_conn_params)
        cur = conn.cursor()

        # CVs
        _process(
            table="cv_embeddings",
            pks=["fonte_aluno", "matricula"],
            emb_col="embedding"
        )

        # Jobs
        _process(
            table="job_embeddings",
            pks=["fonte_aluno", "matricula", "contract_id"],
            emb_col="embedding"
        )

        cur.close()
        conn.close()
        print("✅ shared_latent_code column filled for CVs and Jobs.")