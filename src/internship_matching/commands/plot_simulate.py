"""
plot simulate-tsne-{raw|auto|nomic}

* All background points  … gray (#666)
* Query CV               … blue, size = 6
* Top-K job matches      … red, size = 5, red lines to the query
* Optional --threejs to generate the interactive viewer that re-uses the
  same template already in templates.py (no AxesHelper).

Dependencies already in the project: get_embed_func, CVAutoencoder,
CVJobSharedAutoencoder (if trained), match_jobs_pipeline, THREEJS_TEMPLATE.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, uuid, pickle
from typing import Dict, List, Literal, Tuple

import click
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull          # only for hull of match set
import matplotlib.pyplot as plt

import torch

from internship_matching.data.match import match_jobs_pipeline
from ..constants import COLUMN_SHARED_LATENT_CODE, CV_NOMIC_CLUSTER_FILE_PATH, JOB_NOMIC_CLUSTER_FILE_PATH, JOBS_EMBEDDING_COLUMN_NAME, CVS_EMBEDDING_COLUMN_NAME

from ..data.sharedautoencoder import CVJobSharedAutoencoder

from .root import cli                        # the main Click group
from ..data.jobs import (
    fetch_embeddings_job_with_metadata
)
from ..data.cvs import (
    fetch_single_embedding_cv,
    fetch_single_shared_latent_cv,
)
from ..data.embed import get_embed_func
from ..templates import THREEJS_PLOT_SIM_TEMPLATE, THREEJS_TEMPLATE

PALETTE_GRAY = "#666666"
BLUE   = "#1f77b4"
RED    = "#d62728"

# --------------------------------------------------------------------------- #
def _get_cv_vector(
    cv_file, cv_text, matricula, fonte,
    *,
    embed_func,
    mode: Literal["raw","auto","nomic"]
) -> Tuple[np.ndarray, str]:
    """
    mode == "raw" or "nomic": we need the 768-d nomic embedding
    mode == "auto"            : we need the 96-d shared latent
    """
    # ── TEXT-INPUT CASE ─────────────────────────────────────────────
    if cv_file or cv_text:
        raw = cv_file.read() if cv_file else cv_text
        emb768 = np.array(embed_func(raw), dtype=np.float32)
        if mode == "auto":
            # we *have* to encode it through the AE
            ae = CVJobSharedAutoencoder.load()
            z = ae.encode(emb768[None, :])  # returns shape (1, 96)
            return z.squeeze(0), "CV-text"
        else:
            return emb768, "CV-text"

    # ── MATRICULA CLEAN LOOKUP ───────────────────────────────────────
    if matricula:
        if mode == "auto":
            lat = fetch_single_shared_latent_cv(fonte, matricula)
            if lat is None:
                raise click.ClickException(f"No shared_latent_code for ({fonte},{matricula})")
            return np.asarray(lat, dtype=np.float32), f"{fonte}/{matricula}"
        else:
            vec = fetch_single_embedding_cv(fonte, matricula)
            if vec is None:
                raise click.ClickException(f"No CV embedding for ({fonte},{matricula})")
            return np.asarray(vec, dtype=np.float32), f"{fonte}/{matricula}"

    raise click.ClickException("Provide --cv-file OR --cv-text OR --matricula")

def _tsne_3d(vectors: np.ndarray, perplexity: float = 40.0, seed: int = 42):
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        metric="cosine",
        random_state=seed,
    )
    return tsne.fit_transform(vectors)

# --------------------------------------------------------------------------- #
def _build_threejs_json(
    pts_xyz: np.ndarray,
    is_query: np.ndarray,
    is_match: np.ndarray,
    out_json: Path,
):
    pts: List[Dict[str, object]] = []
    for (x, y, z), q, m in zip(pts_xyz, is_query, is_match):
        if q:
            colour, size = BLUE, 6
        elif m:
            colour, size = RED, 5
        else:
            colour, size = PALETTE_GRAY, 3
        pts.append(
            dict(
                x=float(x),
                y=float(y),
                z=float(z),
                color=colour,
                size=size,
            )
        )

    # convex hull around matches (optional visual aid)
    hull_faces = []
    if is_match.sum() >= 4:
        try:
            hull = ConvexHull(pts_xyz[is_match])
            hull_faces = hull.simplices.tolist()
        except Exception:
            pass

    payload = {
        "points": pts,
        "match_hull": hull_faces,
    }
    out_json.write_text(json.dumps(payload), encoding="utf-8")

# --------------------------------------------------------------------------- #
def _write_html(out_html: Path, title: str, json_fname: str):
    html = THREEJS_PLOT_SIM_TEMPLATE.format(title=title, json_path=json_fname)
    out_html.write_text(html, encoding="utf-8")

def top_k_by_cosine(query: np.ndarray, mat: np.ndarray, k: int):
    mat_n = normalize(mat, axis=1)
    q_n   = normalize(query.reshape(1,-1))[0]
    sims  = mat_n @ q_n
    idx   = np.argsort(sims)[::-1][:k]
    return idx, sims[idx]

# --------------------------------------------------------------------------- #
def _simulate_plot(
    mode: str,
    cv_vec_embedding: np.ndarray,
    job_df,
    *,
    top_k: int,
    threejs: bool,
):
    """
    mode ∈ {"raw", "auto", "nomic"}.
        raw   : cosine in 768-d nomic space
        auto  : cosine in *shared* 96-d latent space
        nomic : uses existing job clusters (centroids) as in compare/run
    """
    click.echo(f"🔍 Starting simulation in '{mode}' mode (top_k={top_k})")

    # ---------- prepare matrices ----------
    click.echo("• Preparing job embedding matrices…")
    job_mat_768 = np.vstack(job_df[JOBS_EMBEDDING_COLUMN_NAME].values).astype(np.float32)
    job_mat_lat = np.vstack(job_df[COLUMN_SHARED_LATENT_CODE].values).astype(np.float32)

    # ---------- select mode & compute initial top_idx ----------
    if mode == "raw":
        click.echo("• Mode = raw (768-d cosine)")
        qv, mat = cv_vec_embedding, job_mat_768
        top_idx, _ = top_k_by_cosine(qv, mat, top_k)
        click.echo(f"  → preliminary top_idx (raw) = {top_idx}")
    elif mode == "auto":
        click.echo("• Mode = auto (shared 96-d latent cosine)")
        qv, mat = cv_vec_embedding, job_mat_lat
        top_idx, _ = top_k_by_cosine(qv, mat, top_k)
        click.echo(f"  → preliminary top_idx (auto) = {top_idx}")
    else:  # nomic
        click.echo("• Mode = nomic (cluster-based match_jobs_pipeline)")
        qv, mat = cv_vec_embedding, job_mat_768
        result = match_jobs_pipeline(
            cv_vec_embedding.squeeze(),
            cv_cluster_file=CV_NOMIC_CLUSTER_FILE_PATH,
            job_cluster_file=JOB_NOMIC_CLUSTER_FILE_PATH,
            cv_centroids_table="cv_nomic_centroids",
            job_centroids_table="job_nomic_centroids",
            job_assignments_table="job_nomic_clusters",
            jobs_fetcher=fetch_embeddings_job_with_metadata,
            embedding_col=JOBS_EMBEDDING_COLUMN_NAME,
            primary_top_k=top_k,
            skip_fit=False,
        )
        matched_contracts = [j["contract_id"] for j in result["matched_jobs"]][:top_k]
        click.echo(f"  → match_jobs_pipeline returned contracts: {matched_contracts}")
        id_to_idx = {cid: i for i, cid in enumerate(job_df["contract_id"])}
        top_idx = [id_to_idx[c] for c in matched_contracts if c in id_to_idx]
        click.echo(f"  → mapped to indices: {top_idx}")

    # ---------- rerank for raw & auto (optional) ----------
    if mode in ("raw", "auto"):
        click.echo("• Re-ranking raw/auto similarities to confirm top_k…")
        sims = normalize(mat, norm="l2") @ normalize(qv.reshape(1, -1))[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        click.echo(f"  → final top_idx = {top_idx}")

    # ---------- TSNE over (jobs + query) ----------
    click.echo("• Running 3-D t-SNE on query + all jobs…")
    pts = np.vstack([qv, mat])  # (N+1, d)
    xyz = _tsne_3d(pts, perplexity=50)
    click.echo("  → t-SNE complete")

    # ---------- build masks ----------
    is_query = np.zeros(len(pts), bool)
    is_query[0] = True
    is_match = np.zeros(len(pts), bool)
    is_match[1 + np.array(top_idx)] = True
    click.echo(f"• Query point at index 0; marking matches at {[1 + i for i in top_idx]}")

    # ---------- write JSON payload ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"sim_{mode}_{ts}"
    out_json = Path("data/processed") / f"{base}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"• Building JSON → {out_json}")
    _build_threejs_json(xyz, is_query, is_match, out_json)
    click.echo(f"✅ JSON saved → {out_json}")

    # ---------- optionally write HTML viewer ----------
    if threejs:
        out_html = Path("data/processed") / f"{base}.html"
        click.echo(f"• Building HTML viewer → {out_html}")
        _write_html(out_html, f"Sim-{mode}", out_json.name)
        click.echo(f"✅ HTML saved → {out_html}")

    # ---------- quick 2D preview plot ----------
    click.echo("• Generating quick 2-D preview plot…")
    fig, ax = plt.subplots(figsize=(5, 5))
    # background
    ax.scatter(
        xyz[~(is_query | is_match), 0],
        xyz[~(is_query | is_match), 1],
        s=3, c=PALETTE_GRAY, alpha=0.4
    )
    # matches
    ax.scatter(xyz[is_match, 0], xyz[is_match, 1], s=3, c=RED)
    # query
    ax.scatter(xyz[is_query, 0], xyz[is_query, 1], s=3, c=BLUE)
    # connecting lines
    for idx in np.where(is_match)[0]:
        ax.plot(
            [xyz[0, 0], xyz[idx, 0]],
            [xyz[0, 1], xyz[idx, 1]],
            c=RED, linewidth=0.4, alpha=0.6
        )
    ax.set_xticks([]); ax.set_yticks([])

    png_path = Path("data/processed") / f"{base}.png"
    plt.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    click.echo(f"✅ 2-D preview saved → {png_path}")

    click.echo("🎉 Simulation complete.")
    
# --------------------------------------------------------------------------- #
#                           CLICK COMMANDS
# --------------------------------------------------------------------------- #
common_opts = [
    click.option('--cv-file', type=click.File('r')),
    click.option('--cv-text', type=str),
    click.option('--matricula', type=str),
    click.option('--fonte', default="VRADM", show_default=True),
    click.option('--top-k', default=20, show_default=True),
    click.option('--threejs/--no-threejs', default=True, show_default=True),
]

def apply_common(f):
    for opt in reversed(common_opts):
        f = opt(f)
    return f

@cli.group()
def plot_sim():
    """Plot simulated t-SNE embeddings of CVs and Jobs."""
    pass

@plot_sim.command("raw")
@apply_common
def plot_sim_raw(cv_file, cv_text, matricula, fonte, top_k, threejs):
    """3-D t-SNE • pure 768-d cosine space."""
    embed_fn = get_embed_func()
    cv_vec, label = _get_cv_vector(cv_file, cv_text, matricula, fonte,
                                   embed_func=embed_fn, mode="raw")
    jobs = fetch_embeddings_job_with_metadata()
    _simulate_plot("raw", cv_vec, jobs, 
                   top_k=top_k, threejs=threejs)

@plot_sim.command("auto")
@apply_common
def plot_sim_auto(cv_file, cv_text, matricula, fonte, top_k, threejs):
    """3-D t-SNE • shared auto-encoder latent cosine."""
    embed_fn = get_embed_func()
    cv_vec, label = _get_cv_vector(cv_file, cv_text, matricula, fonte,
                                   embed_func=embed_fn, mode="auto")
    jobs = fetch_embeddings_job_with_metadata()
    _simulate_plot("auto", cv_vec, jobs, 
                   top_k=top_k, threejs=threejs)

@plot_sim.command("nomic")
@apply_common
def plot_sim_nomic(cv_file, cv_text, matricula, fonte, top_k, threejs):
    """3-D t-SNE • nomic-cluster-based match set (red) vs query (blue)."""
    embed_fn = get_embed_func()
    cv_vec, label = _get_cv_vector(cv_file, cv_text, matricula, fonte,
                                   embed_func=embed_fn, mode="nomic")
    jobs = fetch_embeddings_job_with_metadata()
    _simulate_plot("nomic", cv_vec, jobs, 
                   top_k=top_k, threejs=threejs)