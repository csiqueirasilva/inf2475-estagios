import json
import math
import pathlib
from typing import Any, Callable, Dict, Union
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
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

from ..constants import CVS_AUTOENCODER_FILE_PATH, CVS_AUTOENCODER_LOSS_TYPE, CVS_EMBEDDING_COLUMN_NAME, CVS_HIDDEN_SIZE, CVS_LATENT_SIZE, CVS_ORIGINAL_SIZE, CVS_TRAIN_SEED

from .db import POSTGRES_URL
from .jobs import TORCH_DEVICE
from .cvs import fetch_embeddings_cv, fetch_single_embedding_cv

class CVAutoencoder(nn.Module):
    """
    Autoencoder for CV embeddings.
    Moves itself to the configured TORCH_DEVICE on instantiation.

    Provides convenient classmethods for loading and training.
    """
    def __init__(
        self,
        input_dim: int = CVS_ORIGINAL_SIZE,
        hidden_dim: int = CVS_HIDDEN_SIZE,
        latent_dim: int = CVS_LATENT_SIZE
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
        # Move model to configured device
        self.to(TORCH_DEVICE)

    def save_round_trip_report(
        self,
        embeds: np.ndarray,
        path: Union[str, pathlib.Path]
    ) -> None:
        """
        Computes encode→decode and decode→encode MAEs on `embeds`,
        and writes a JSON report to `path` with min/mean/max
        (and raw arrays if you ever want them).
        """
        # make sure model is on the right device & in eval mode
        self.eval()
        device = next(self.parameters()).device

        # get reconstructions & errors
        recons = self.get_reconstructions(embeds)
        enc_dec_mae, dec_enc_mae = self.round_trip_errors(embeds)

        report = {
            "encode_decode_mae": {
                "min": float(enc_dec_mae.min()),
                "mean": float(enc_dec_mae.mean()),
                "max": float(enc_dec_mae.max()),
                # "all": enc_dec_mae.tolist()   # uncomment if you want the full array
            },
            "decode_encode_mae": {
                "min": float(dec_enc_mae.min()),
                "mean": float(dec_enc_mae.mean()),
                "max": float(dec_enc_mae.max()),
                # "all": dec_enc_mae.tolist()
            }
        }

        # write JSON file
        with open(path, "w") as fp:
            json.dump(report, fp, indent=2)

        print(f"✅ Round-trip MAE report saved to {path!s}")

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): shape (batch_size, input_dim)

        Returns:
            recon (torch.Tensor): shape (batch_size, input_dim)
            z     (torch.Tensor): shape (batch_size, latent_dim)
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def round_trip_errors(
        self,
        embeds: np.ndarray,                     # shape (N, input_dim)
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For each x in `embeds`, compute two errors:
          1) enc–dec error   =  || decode(encode(x)) – x ||
          2) dec–enc error   =  || encode(decode(encode(x))) – encode(x) ||

        Returns:
            enc_dec_mae (np.ndarray): shape (N,), mean abs error per sample in input space
            dec_enc_mae (np.ndarray): shape (N,), mean abs error per sample in latent space
        """
        self.eval()
        self.to(device)

        tensor = torch.from_numpy(embeds).to(device)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

        enc_dec_errs = []
        dec_enc_errs = []

        with torch.no_grad():
            for (batch,) in loader:
                # 1) encode → decode
                recon, z = self(batch)                   # recon: input‐space, z: latent‐space
                enc_dec_errs.append((recon - batch).cpu().numpy())

                # 2) decode → encode
                #    re‐encode the reconstruction, compare its code to the original z
                _, z2 = self(recon)
                dec_enc_errs.append((z2 - z).cpu().numpy())

        enc_dec_errs = np.vstack(enc_dec_errs)           # shape (N, input_dim)
        dec_enc_errs = np.vstack(dec_enc_errs)           # shape (N, latent_dim)

        # mean absolute error per sample
        enc_dec_mae = np.abs(enc_dec_errs).mean(axis=1)  # (N,)
        dec_enc_mae = np.abs(dec_enc_errs).mean(axis=1)  # (N,)

        return enc_dec_mae, dec_enc_mae

    def get_reconstructions(
        self,
        embeds: np.ndarray,                    # shape (N, input_dim)
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE
    ) -> np.ndarray:
        """
        Given an array of input embeddings, return the autoencoder's reconstructions.

        Args:
            embeds (np.ndarray): raw input embeddings, shape (N, input_dim)
            batch_size (int): how many samples per forward-pass
            device (torch.device): model & data device

        Returns:
            np.ndarray: reconstructed embeddings, shape (N, input_dim)
        """
        # prepare
        self.eval()
        self.to(device)
        tensor = torch.from_numpy(embeds).to(device)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

        # forward‐pass in batches
        recon_list = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                recon, _ = self(batch)
                recon_list.append(recon.cpu().numpy())

        # stack back into (N, input_dim)
        return np.vstack(recon_list)

    @classmethod
    def load(
        cls,
        path: str,
        input_dim: int = CVS_ORIGINAL_SIZE,
        hidden_dim: int = CVS_HIDDEN_SIZE,
        latent_dim: int = CVS_LATENT_SIZE,
        device: torch.device = TORCH_DEVICE
    ) -> "CVAutoencoder":
        """
        Instantiate and load a trained autoencoder from disk.

        Returns:
            CVAutoencoder: model in eval mode on the given device
        """
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
        hidden_dim: int = CVS_HIDDEN_SIZE,
        latent_dim: int = CVS_LATENT_SIZE,
        train_frac: float = 0.8,
        train_seed: int = CVS_TRAIN_SEED,
        loss_type: str = CVS_AUTOENCODER_LOSS_TYPE  # options: "mse", "cosine", "hybrid"
    ) -> "CVAutoencoder":
        """
        Fetch embeddings, split train/validation reproducibly, train and save.

        Args:
            epochs (int): number of passes over the data
            batch_size (int): samples per batch
            lr (float): learning rate
            hidden_dim (int): size of hidden layer
            latent_dim (int): size of latent code
            train_frac (float): fraction of data for training
            train_seed (int): RNG seed for split
            loss_type (str): "mse" (mean-squared), "cosine" (angular), or "hybrid" (sum of both)
        """
        device = TORCH_DEVICE

        # Load & split data
        embeds = fetch_embeddings_cv()  # (N, D)
        tensor = torch.from_numpy(embeds).to(device)
        full_ds = TensorDataset(tensor)
        total = len(full_ds)
        train_n = int(train_frac * total)
        val_n = total - train_n
        generator = torch.Generator().manual_seed(train_seed)
        train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=generator)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Model & optimizer
        model = cls(input_dim=embeds.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            train_mse = train_cos = 0.0

            for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
                batch = batch.to(device)
                recon, _ = model(batch)

                # metrics
                mse_loss = F.mse_loss(recon, batch, reduction="mean")
                cos_loss = 1 - F.cosine_similarity(recon, batch, dim=1).mean()

                # select loss
                if loss_type == "mse":
                    loss = mse_loss
                elif loss_type == "cosine":
                    loss = cos_loss
                elif loss_type == "hybrid":
                    loss = mse_loss + cos_loss
                else:
                    raise ValueError("loss_type must be 'mse', 'cosine', or 'hybrid'")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_mse += mse_loss.item() * batch.size(0)
                train_cos += cos_loss.item() * batch.size(0)

            avg_train_mse = train_mse / train_n
            avg_train_cos = train_cos / train_n

            # Validation
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

        # Save
        torch.save(model.state_dict(), CVS_AUTOENCODER_FILE_PATH)
        print(
            f"Trained (loss_type={loss_type}, seed={train_seed}, "
            f"{train_frac*100:.0f}% train) saved to {CVS_AUTOENCODER_FILE_PATH}"
        )
        return model

    @classmethod
    def plot_latent_space_with_categories_facet(
        cls,
        model: "CVAutoencoder",
        embeds: np.ndarray,                   # shape (N, D)
        reduction: str = "pca",               # "pca", "tsne" or "umap"
        labels: np.ndarray | None = None,     # integer array (N,) 0..C-1
        categories: list[str] | None = None,  # list of C names
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE,
        output_path: str = "latent_space.pdf",
        show: bool = False,
        seed: int = CVS_TRAIN_SEED
    ):
        """
        Faceted‐highlight plot: one small subplot per category in `categories`,
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
                latents.append(z.cpu().numpy())
        Z = np.vstack(latents)  # (N, latent_dim)

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

    @classmethod
    def plot_latent_space_with_categories(
        cls,
        model: "CVAutoencoder",
        embeds: np.ndarray,                   # shape (N, D)
        reduction: str = "pca",               # "pca", "tsne" or "umap"
        labels: np.ndarray | None = None,     # optional array of length N for coloring
        categories: list[str] | None = None,  # optional list of category names
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE,
        output_path: str = "cv_latent_space_$reduction$.png",
        dpi: int = 300,
        show: bool = False,
        seed : int = CVS_TRAIN_SEED,
        latent_size : int = CVS_LATENT_SIZE,
        embedding_column : str = CVS_EMBEDDING_COLUMN_NAME,
        loss_type : str = CVS_AUTOENCODER_LOSS_TYPE,
        plot_updates : Callable = None
    ):
        model.eval()

        # 1) Encode to latent space
        tensor = torch.from_numpy(embeds.astype(np.float32)).to(device)
        loader = DataLoader(TensorDataset(tensor),
                            batch_size=batch_size,
                            shuffle=False)
        latents = []
        with torch.no_grad():
            for (batch,) in loader:
                _, z = model(batch)
                latents.append(z.cpu().numpy())
        Z = np.vstack(latents)   # (N, latent_dim)

        # 2) Reduce
        r = reduction.lower()
        if r == "pca":
            reducer = PCA(n_components=2, random_state=seed)
        elif r == "tsne":
            reducer = TSNE(
                n_components=2,
                init="pca",
                random_state=seed,
                learning_rate="auto",
                perplexity=50
            )
        elif r == "umap":
            reducer = UMAP(
                n_components=2,
                n_neighbors=50,
                min_dist=0.0,
                random_state=seed
            )
        else:
            raise ValueError("reduction must be 'pca', 'tsne' or 'umap'")
        Z2 = reducer.fit_transform(Z)

        # 3) Plot
        plt.figure(figsize=(6,6))
        scatter_kwargs = dict(s=5, alpha=0.7)

        if labels is not None:
            if categories is not None:
                # discrete categorical colouring
                n_cat = len(categories)
                cmap = plt.cm.get_cmap("hsv", n_cat)
                # boundary norm so each integer maps to its bin
                norm = BoundaryNorm(np.arange(n_cat+1)-0.5, n_cat)
                scatter_kwargs.update(c=labels, cmap=cmap, norm=norm)
            else:
                # continuous colouring
                scatter_kwargs.update(c=labels, cmap="viridis")

        plt.scatter(Z2[:,0], Z2[:,1], **scatter_kwargs)

        # allow any extra tweaks (e.g. your add_course_colorbar)
        if plot_updates is not None:
            plot_updates(plt)

        # legend vs colorbar
        if labels is not None:
            if categories is not None:
                # build discrete legend
                handles = [
                    Line2D([0],[0],
                           marker="o",
                           color=plt.cm.hsv(i / len(categories)),
                           linestyle="",
                           markersize=6)
                    for i in range(len(categories))
                ]
                plt.legend(
                    handles,
                    categories,
                    loc="center left",
                    bbox_to_anchor=(1.0, 0.5),
                    title="Category",
                    fontsize="small",
                    title_fontsize="small"
                )
            else:
                plt.colorbar(label="label / value")

        plt.title(f"Latent → 2D ({r.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()

        # 4) Save / show
        save_path = (
            output_path
            .replace("$reduction$", r)
            .replace("$latent_size$", str(latent_size))
            .replace("$embedding_column$", embedding_column)
            .replace("$seed_number$", str(seed))
            .replace("$loss_type$", loss_type)
        )
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        print(f"Latent‐space plot saved to {save_path}")


    @classmethod
    def plot_latent_space(
        cls,
        model: "CVAutoencoder",
        embeds: np.ndarray,                   # shape (N, D)
        reduction: str = "pca",               # "pca", "tsne" or "umap"
        labels: np.ndarray | None = None,     # optional array of length N for coloring
        batch_size: int = 256,
        device: torch.device = TORCH_DEVICE,
        output_path: str = "data/processed/cv_latent_space_$reduction$_$latent_size$_$embedding_column$_$seed_number$_$loss_type$.png",
        dpi: int = 300,
        show: bool = False,
        seed : int = CVS_TRAIN_SEED,
        latent_size : int = CVS_LATENT_SIZE,
        embedding_column : str = CVS_EMBEDDING_COLUMN_NAME,
        loss_type : str = CVS_AUTOENCODER_LOSS_TYPE,
        plot_updates : Callable = None
    ):
        """
        Project the autoencoder's latent codes to 2D and save as PNG.

        Args:
            model:       a trained CVAutoencoder
            embeds:      raw embeddings array of shape (N, D)
            reduction:   one of "pca", "tsne", or "umap"
            labels:      optional array (N,) to color each point (e.g. class IDs or errors)
            batch_size:  samples per batch for encoding
            device:      torch device
            output_path: filesystem path with `$reduction$` placeholder
            dpi:         resolution for saved figure
            show:        if True, also call plt.show()
        """
        model.eval()

        # 1) Encode into latent space
        tensor = torch.from_numpy(embeds.astype(np.float32)).to(device)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

        latents = []
        with torch.no_grad():
            for (batch,) in loader:
                _, z = model(batch)
                latents.append(z.cpu().numpy())
        Z = np.vstack(latents)   # (N, latent_dim)

        # 2) Choose & fit reducer
        r = reduction.lower()
        if r == "pca":
            reducer = PCA(n_components=2, random_state=seed)
        elif r == "tsne":
            reducer = TSNE(n_components=2, init="pca", random_state=seed, learning_rate="auto", perplexity=50)
        elif r == "umap":
            reducer = UMAP(n_components=2, n_neighbors=50, min_dist=0, random_state=seed)
        else:
            raise ValueError("reduction must be one of 'pca', 'tsne', or 'umap'")

        Z2 = reducer.fit_transform(Z)   # (N, 2)

        # 3) Plot
        plt.figure(figsize=(6,6))
        scatter_kwargs = dict(s=5, alpha=0.7)
        if labels is not None:
            scatter_kwargs.update(c=labels, cmap="viridis")
        plt.scatter(Z2[:,0], Z2[:,1], **scatter_kwargs)

        if plot_updates is not None:
            plot_updates(plt)

        # colorbar for continuous labels
        if labels is not None:
            plt.colorbar(label="label / value")

        plt.title(f"Latent space → 2D ({r.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()

        # 4) Save & show
        save_path = output_path.replace("$reduction$", r)
        save_path = save_path.replace("$latent_size$", str(latent_size))
        save_path = save_path.replace("$embedding_column$", str(embedding_column))
        save_path = save_path.replace("$seed_number$", str(seed))
        save_path = save_path.replace("$loss_type$", str(loss_type))
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        print(f"Latent-space plot saved to {save_path}")

    @classmethod
    def test_cv(
        cls,
        matricula: str,
        fonte_aluno: str = "VRADM"
    ) -> dict:
        """
        Load the trained autoencoder, fetch a single CV embedding, and compute
        its latent code and reconstruction MSE.

        Args:
            matricula (str): student identifier
            fonte_aluno (str): data source key (default "VRADM")

        Returns:
            dict: {
                "latent": List[float],
                "mse": float
            }
        """
        # Load model
        model = cls.load(CVS_AUTOENCODER_FILE_PATH)
        device = TORCH_DEVICE
        # Fetch embedding
        vec = fetch_single_embedding_cv(fonte_aluno, matricula)
        if vec is None:
            raise ValueError(f"No embedding found for ({fonte_aluno}, {matricula})")
        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        # Run inference
        with torch.no_grad():
            recon, z = model(x)
            mse = nn.functional.mse_loss(recon, x).item()
        return {"latent": z.squeeze(0).tolist(), "mse": mse}

    @classmethod
    def generate_all_latents(
        cls,
        pg_conn_params: Union[str, Dict[str, Any]] = "postgresql://postgres:postgres@localhost:5432/ai_test",
        batch_size: int = 64,
        input_dim: int = CVS_ORIGINAL_SIZE,
        hidden_dim: int = CVS_HIDDEN_SIZE,
        latent_dim: int = CVS_LATENT_SIZE,
        device: torch.device = TORCH_DEVICE        
    ) -> None:
        """
        Fetches all embeddings from the database, computes latent codes in batches,
        and upserts them into the latent_code column.

        Args:
            pg_conn_params (str|dict): Postgres DSN string or connection parameters.
            batch_size (int): Number of embeddings to process per batch.
        """
        # Load trained autoencoder
        ae = cls.load(CVS_AUTOENCODER_FILE_PATH, input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        ae.eval()

        # Connect to Postgres
        conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
               else psycopg2.connect(**pg_conn_params)
        cur = conn.cursor()

        # Count total entries
        cur.execute("SELECT COUNT(*) FROM cv_embeddings WHERE embedding IS NOT NULL")
        total = cur.fetchone()[0]

        # Pagination over records with embedding
        offset = 0
        with tqdm(total=total, desc="Generating Latents", unit="cv") as pbar:
            while offset < total:
                cur.execute(
                    "SELECT fonte_aluno, matricula, embedding FROM cv_embeddings WHERE embedding IS NOT NULL ORDER BY fonte_aluno, matricula LIMIT %s OFFSET %s",
                    (batch_size, offset)
                )
                rows = cur.fetchall()
                if not rows:
                    break

                                # Extract embeddings and identifiers
                ids = [(r[0], r[1]) for r in rows]
                raw_embeds = [r[2] for r in rows]
                # Parse embedding strings into lists if needed
                embeds = []
                import json, ast
                for e in raw_embeds:
                    if isinstance(e, str):
                        try:
                            embeds.append(json.loads(e))
                        except json.JSONDecodeError:
                            embeds.append(ast.literal_eval(e))
                    else:
                        embeds.append(e)

                # Compute latents
                emb_t = torch.tensor(embeds, dtype=torch.float32, device=device)
                with torch.no_grad():
                    _, latents = ae(emb_t)
                lat_list = latents.cpu().tolist()

                # Upsert latent_code for each record
                for (fonte, matric), lat in zip(ids, lat_list):
                    cur.execute(
                        "UPDATE cv_embeddings SET latent_code = %s WHERE fonte_aluno = %s AND matricula = %s",
                        (lat, fonte, matric)
                    )
                    pbar.update(1)

                conn.commit()
                offset += len(rows)

        cur.close()
        conn.close()

        print("All latent codes generated and upserted.")
        return None