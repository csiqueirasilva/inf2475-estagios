from pathlib import Path
from typing import Dict

SHARED_AE_FILE_PATH = "data/processed/shared_autoencoder.pt"
SHARED_HIDDEN_SIZE = 128
SHARED_LATENT_SIZE = 96

CVS_AUTOENCODER_FILE_PATH = "data/processed/cv_autoencoder.pt"

CVS_EMBEDDING_COLUMN_NAME = "embedding"

CVS_AUTOENCODER_LOSS_TYPE = "mse"

GLOBAL_RANDOM_SEED = 42

CVS_ORIGINAL_SIZE = 768
CVS_HIDDEN_SIZE = 128
CVS_TRAIN_SEED = GLOBAL_RANDOM_SEED

CVS_LATENT_SIZE = 96

JOBS_AUTOENCODER_FILE_PATH = "data/processed/job_autoencoder.pt"
JOBS_AUTOENCODER_LOSS_TYPE = "mse"
JOBS_EMBEDDING_COLUMN_NAME = "embedding"
JOBS_HIDDEN_SIZE = 128
JOBS_LATENT_SIZE = 96
JOBS_ORIGINAL_SIZE = 768
JOBS_TRAIN_SEED = GLOBAL_RANDOM_SEED

CV_CLUSTER_FILE_PATH = "data/processed/cv_hdbscan_clusters.pkl"
JOB_CLUSTER_FILE_PATH = "data/processed/job_hdbscan_clusters.pkl"

CV_NOMIC_CLUSTER_FILE_PATH = "data/processed/cv_nomic_hdbscan_clusters.pkl"
JOB_NOMIC_CLUSTER_FILE_PATH = "data/processed/job_nomic_hdbscan_clusters.pkl"

DEFAULT_DATABASE_FILE = "data/processed/data.db"

COLUMN_LATENT_CODE = "latent_code"
COLUMN_SHARED_LATENT_CODE = "shared_latent_code"

CLUSTERER_PATHS: Dict[str, Path] = {
    "job-nomic":      JOB_NOMIC_CLUSTER_FILE_PATH,
    "job-autoencode": JOB_CLUSTER_FILE_PATH,
    "cv-nomic":       CV_NOMIC_CLUSTER_FILE_PATH,
    "cv-autoencode":  CV_CLUSTER_FILE_PATH,
}

DEFAULT_TOP_K_LABELS = 50

DEFAULT_PIPELINE_TOP_K_LABELS = 1