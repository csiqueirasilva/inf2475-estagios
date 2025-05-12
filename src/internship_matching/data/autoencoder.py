from typing import Any, Dict, Union
import psycopg2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .db import POSTGRES_URL
from .jobs import TORCH_DEVICE
from .cvs import fetch_embeddings_cv, fetch_single_embedding_cv

CVS_AUTOENCODER_FILE_PATH = "data/processed/cv_autoencoder.pt"

class CVAutoencoder(nn.Module):
    """
    Autoencoder for CV embeddings.
    Moves itself to the configured TORCH_DEVICE on instantiation.

    Provides convenient classmethods for loading and training.
    """
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        latent_dim: int = 16
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

    @classmethod
    def load(
        cls,
        path: str,
        input_dim: int = 768,
        hidden_dim: int = 128,
        latent_dim: int = 16,
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
        hidden_dim: int = 128,
        latent_dim: int = 16
    ) -> "CVAutoencoder":
        """
        Fetch embeddings from Postgres, train autoencoder, and save the model.

        Args:
            epochs (int): number of passes over the data
            batch_size (int): samples per batch
            lr (float): learning rate
            hidden_dim (int): size of hidden layer
            latent_dim (int): size of latent code

        Returns:
            CVAutoencoder: trained model
        """
        # Device setup
        device = TORCH_DEVICE
        # Load data
        embeds = fetch_embeddings_cv()
        tensor = torch.from_numpy(embeds).to(device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Model, loss, optimizer
        model = cls(input_dim=embeds.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            with tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as bar:
                for (batch,) in bar:
                    batch = batch.to(device)
                    recon, _ = model(batch)
                    loss = criterion(recon, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch.size(0)
                    bar.set_postfix(loss=loss.item())
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch}/{epochs} completed. Avg Loss: {avg_loss:.6f}")

        # Save model
        torch.save(model.state_dict(), CVS_AUTOENCODER_FILE_PATH)
        print(f"Autoencoder trained and saved to {CVS_AUTOENCODER_FILE_PATH}")
        return model

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
        batch_size: int = 64
    ) -> None:
        """
        Fetches all embeddings from the database, computes latent codes in batches,
        and upserts them into the latent_code column.

        Args:
            pg_conn_params (str|dict): Postgres DSN string or connection parameters.
            batch_size (int): Number of embeddings to process per batch.
        """
        # Load trained autoencoder
        device = TORCH_DEVICE
        ae = cls.load(CVS_AUTOENCODER_FILE_PATH)
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