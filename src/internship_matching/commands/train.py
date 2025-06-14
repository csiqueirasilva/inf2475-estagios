
from .root import cli
from ..utils import deprecated
from ..data.job_autoencoder import JobAutoencoder
from ..data.autolabeler import CVAutoLabeler
from ..data.autoencoder import CVAutoencoder
from ..training.train_cv_job    import train_cv_job_matching
from ..training.train_cv_feat   import train_cv_feature_scoring
from ..training.train_job_feat  import train_job_feature_scoring

# ─── TRAIN GROUP ──────────────────────────────────────────────────────────────
@cli.group()
def train():
    """Model training commands."""
    pass

@train.command("job-autoencoder")
def train_cv_autoencoder():
    JobAutoencoder.train_from_db()
    JobAutoencoder.generate_all_latents()

@train.command("cv-autoencoder")
def train_cv_autoencoder():
    CVAutoencoder.train_from_db()
    CVAutoencoder.generate_all_latents()

@deprecated("Foi um experimento, não é usado no trabalho. Acabei usando diversas tecnologias para testar o conceito.")
@train.command("cv-autolabeler")
def train_cv_autolabeler():
    labeler = CVAutoLabeler()
    # labeler.fit_kmeans()
    # labeler.name_clusters_keybert()
    # labeler.save_keybert()
    # labeler.name_clusters_tfidf()
    # labeler.save_tfidf()
    # labeler.name_clusters_ctfidf()
    # labeler.save_ctfidf()
    labeler.name_clusters_bertopic()
    labeler.save_bertopic()

@deprecated("Acabou sendo substituído pelo comando de treinar os clusters individualmente.")
@train.command("cv-job")
def train_cv_job():
    train_cv_job_matching()

@deprecated("Não foi possível extrair features dos CVs.")
@train.command("cv-feat")
def train_cv_feat():
    train_cv_feature_scoring()

@deprecated("Não foi possível extrair features dos Jobs.")
@train.command("job-feat")
def train_job_feat():
    train_job_feature_scoring()
