import json
import math
import pathlib
from typing import Any, Callable, Dict, Union
import numpy as np
import psycopg2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..constants import (
    JOBS_AUTOENCODER_FILE_PATH,
    JOBS_AUTOENCODER_LOSS_TYPE,
    JOBS_EMBEDDING_COLUMN_NAME,
    JOBS_HIDDEN_SIZE,
    JOBS_LATENT_SIZE,
    JOBS_ORIGINAL_SIZE,
    JOBS_TRAIN_SEED,
)
from .db import POSTGRES_URL
from .jobs import TORCH_DEVICE, fetch_embeddings_job, fetch_single_embedding_job


class JobAutoencoder(nn.Module):
    """
    Autoencoder for job embeddings.
    Moves itself to the configured TORCH_DEVICE on instantiation.

    Provides convenient classmethods for loading and training.
    """
    def __init__(
        self,
        input_dim: int = JOBS_ORIGINAL_SIZE,
        hidden_dim: int = JOBS_HIDDEN_SIZE,
        latent_dim: int = JOBS_LATENT_SIZE,
    ):
        super().__init__()
        # Encoder: input_dim -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        # Decoder: latent_dim -> hidden_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )
        self.to(TORCH_DEVICE)

    def save_round_trip_report(
        self,
        embeds: np.ndarray,
        path: Union[str, pathlib.Path],
    ) -> None:
        self.eval()
        device = next(self.parameters()).device

        recons = self.get_reconstructions(embeds)
        enc_dec_mae, dec_enc_mae = self.round_trip_errors(embeds)

        report = {
            "encode_decode_mae": {
                "min": float(enc_dec_mae.min()),
                "mean": float(enc_dec_mae.mean()),
                "max": float(enc_dec_mae.max()),
            },
            "decode_encode_mae": {
                "min": float(dec_enc_mae.min()),
                "mean": float(dec_enc_mae.mean()),
                "max": float(dec_enc_mae.max()),
            },
        }

        with open(path, "w") as fp:
            json.dump(report, fp, indent=2)

        print(f"✅ Round-trip MAE report saved to {path}")

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def round_trip_errors(
        self,
        embeds: np.ndarray,
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        self.to(device)
        tensor = torch.from_numpy(embeds.astype(np.float32)).to(device)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

        enc_dec_errs = []
        dec_enc_errs = []
        with torch.no_grad():
            for (batch,) in loader:
                recon, z = self(batch)
                enc_dec_errs.append((recon - batch).cpu().numpy())
                _, z2 = self(recon)
                dec_enc_errs.append((z2 - z).cpu().numpy())

        enc_dec_errs = np.vstack(enc_dec_errs)
        dec_enc_errs = np.vstack(dec_enc_errs)
        enc_dec_mae = np.abs(enc_dec_errs).mean(axis=1)
        dec_enc_mae = np.abs(dec_enc_errs).mean(axis=1)
        return enc_dec_mae, dec_enc_mae

    def get_reconstructions(
        self,
        embeds: np.ndarray,
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE,
    ) -> np.ndarray:
        self.eval()
        self.to(device)
        tensor = torch.from_numpy(embeds.astype(np.float32)).to(device)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

        recon_list = []
        with torch.no_grad():
            for (batch,) in loader:
                recon, _ = self(batch)
                recon_list.append(recon.cpu().numpy())
        return np.vstack(recon_list)

    @classmethod
    def load(
        cls,
        path: str,
        input_dim: int = JOBS_ORIGINAL_SIZE,
        hidden_dim: int = JOBS_HIDDEN_SIZE,
        latent_dim: int = JOBS_LATENT_SIZE,
        device: torch.device = TORCH_DEVICE,
    ) -> "JobAutoencoder":
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    @classmethod
    def train_from_db(
        cls,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        hidden_dim: int = JOBS_HIDDEN_SIZE,
        latent_dim: int = JOBS_LATENT_SIZE,
        train_frac: float = 0.8,
        train_seed: int = JOBS_TRAIN_SEED,
        loss_type: str = JOBS_AUTOENCODER_LOSS_TYPE,
    ) -> "JobAutoencoder":
        device = TORCH_DEVICE
        embeds = fetch_embeddings_job()
        tensor = torch.from_numpy(embeds.astype(np.float32)).to(device)
        full_ds = TensorDataset(tensor)
        total = len(full_ds)
        train_n = int(train_frac * total)
        val_n = total - train_n
        gen = torch.Generator().manual_seed(train_seed)
        train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=gen)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = cls(input_dim=embeds.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            train_mse = train_cos = 0.0
            for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
                batch = batch.to(device)
                recon, _ = model(batch)
                mse_loss = F.mse_loss(recon, batch, reduction="mean")
                cos_loss = 1 - F.cosine_similarity(recon, batch, dim=1).mean()
                if loss_type == "mse":
                    loss = mse_loss
                elif loss_type == "cosine":
                    loss = cos_loss
                else:
                    loss = mse_loss + cos_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_mse += mse_loss.item() * batch.size(0)
                train_cos += cos_loss.item() * batch.size(0)

            avg_train_mse = train_mse / train_n
            avg_train_cos = train_cos / train_n
            model.eval()
            val_mse = val_cos = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    recon, _ = model(batch)
                    mse_loss = F.mse_loss(recon, batch, reduction="mean")
                    cos_loss = 1 - F.cosine_similarity(recon, batch, dim=1).mean()
                    val_mse += mse_loss.item() * batch.size(0)
                    val_cos += cos_loss.item() * batch.size(0)

            avg_val_mse = val_mse / val_n
            avg_val_cos = val_cos / val_n
            print(
                f"Epoch {epoch}/{epochs} — "
                f"Train[MSE={avg_train_mse:.6f}, CosDist={avg_train_cos:.6f}] | "
                f"Val[MSE={avg_val_mse:.6f}, CosDist={avg_val_cos:.6f}]"
            )

        torch.save(model.state_dict(), JOBS_AUTOENCODER_FILE_PATH)
        print(
            f"Trained (loss_type={loss_type}, seed={train_seed}, "
            f"{train_frac*100:.0f}% train) saved to {JOBS_AUTOENCODER_FILE_PATH}"
        )
        return model

    @classmethod
    def generate_all_latents(
        cls,
        pg_conn_params: Union[str, Dict[str, Any]] = POSTGRES_URL,
        batch_size: int = 64,
        input_dim: int = JOBS_ORIGINAL_SIZE,
        hidden_dim: int = JOBS_HIDDEN_SIZE,
        latent_dim: int = JOBS_LATENT_SIZE,
        device: torch.device = TORCH_DEVICE,
    ) -> None:
        ae = cls.load(JOBS_AUTOENCODER_FILE_PATH, input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        ae.eval()
        conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) else psycopg2.connect(**pg_conn_params)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM job_embeddings WHERE embedding IS NOT NULL")
        total = cur.fetchone()[0]
        offset = 0
        with tqdm(total=total, desc="Generating Latents", unit="job") as pbar:
            while offset < total:
                cur.execute(
                    "SELECT fonte_aluno, matricula, contract_id, embedding FROM job_embeddings WHERE embedding IS NOT NULL ORDER BY fonte_aluno, matricula, contract_id LIMIT %s OFFSET %s",
                    (batch_size, offset),
                )
                rows = cur.fetchall()
                if not rows:
                    break
                ids = [(r[0], r[1], r[2]) for r in rows]
                raw_embeds = [r[3] for r in rows]
                embeds = []
                import ast
                for e in raw_embeds:
                    if isinstance(e, str):
                        try:
                            embeds.append(json.loads(e))
                        except json.JSONDecodeError:
                            embeds.append(ast.literal_eval(e))
                    else:
                        embeds.append(e)
                emb_t = torch.tensor(np.array(embeds, dtype=np.float32), device=device)
                with torch.no_grad():
                    _, latents = ae(emb_t)
                lat_list = latents.cpu().tolist()
                for (fonte, matric, cid), lat in zip(ids, lat_list):
                    cur.execute(
                        "UPDATE job_embeddings SET latent_code = %s WHERE fonte_aluno = %s AND matricula = %s AND contract_id = %s",
                        (lat, fonte, matric, cid),
                    )
                    pbar.update(1)
                conn.commit()
                offset += len(rows)
        cur.close()
        conn.close()
        print("All latent codes generated and upserted.")

    @classmethod
    def plot_latent_space_with_categories_facet(
        cls,
        model: "JobAutoencoder",
        embeds: np.ndarray,
        reduction: str = "pca",
        labels: np.ndarray | None = None,
        categories: list[str] | None = None,
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE,
        output_path: str = "latent_space.pdf",
        show: bool = False,
        seed: int = JOBS_TRAIN_SEED,
    ):
        """
        Faceted-highlight plot: one small subplot per category in `categories`,
        showing that category in color vs. all others in gray.
        """
        # ─── 0) Seed ALL the things ───────────────────────────────────────────────
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Make CuDNN deterministic (may slow things down a bit)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

        model.eval()

        # ─── 1) Encode into latent space ────────────────────────────────────────
        tensor = torch.from_numpy(embeds).to(device)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
        latents = []
        with torch.no_grad():
            for (batch,) in loader:
                _, z = model(batch)
                latents.append(z)
        Z = torch.cat(latents, dim=0).cpu().numpy()  # (N, latent_dim)

        # ─── 2) Reduce to 2D ────────────────────────────────────────────────────
        r = reduction.lower()
        if r == "pca":
            # PCA is already deterministic when using full SVD,
            # but we pass random_state for consistency if you switch solvers.
            reducer = PCA(n_components=2, random_state=seed)

        elif r == "tsne":
            # TSNE(random_state) makes the embedding deterministic
            reducer = TSNE(
                n_components=2,
                init="pca",
                random_state=seed,
                learning_rate="auto",
                perplexity=50,
                metric="cosine"
            )

        elif r == "umap":
            # UMAP(random_state + init="pca") is also reproducible.
            reducer = UMAP(
                n_components=2,
                n_neighbors=10,
                min_dist=1.0,
                metric="cosine",
                init="pca",
                random_state=seed
            )

        else:
            raise ValueError("reduction must be one of 'pca', 'tsne', or 'umap'")

        Z2 = reducer.fit_transform(Z)  # (N, 2)

        # ─── 3) Set up facet grid ───────────────────────────────────────────────
        C = len(categories or [])
        if C == 0:
            raise ValueError("Must supply a non-empty categories list")
        ncols = 4
        nrows = math.ceil(C / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(4 * ncols, 4 * nrows),
                                squeeze=False)

        # ─── 4) Draw each facet ────────────────────────────────────────────────
        for idx, cat in enumerate(categories):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            mask = (labels == idx)
            # background: everyone else in light gray
            ax.scatter(
                Z2[~mask, 0], Z2[~mask, 1],
                c="lightgray", s=5, alpha=0.2, rasterized=True
            )
            # highlight this category
            ax.scatter(
                Z2[mask, 0], Z2[mask, 1],
                c=[plt.cm.tab20(idx % 20)],
                s=20, alpha=0.8,
                edgecolor="k", linewidth=0.3, rasterized=True
            )
            ax.set_title(cat, fontsize="small")
            ax.set_xticks([]); ax.set_yticks([])

        # ─── 5) Hide unused subplots ────────────────────────────────────────────
        for extra in range(C, nrows * ncols):
            r0, c0 = divmod(extra, ncols)
            axes[r0][c0].axis("off")

        plt.tight_layout()

        # ─── 6) Save & show ────────────────────────────────────────────────────
        plt.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        print(f"Faceted highlight plot saved to {output_path}")
