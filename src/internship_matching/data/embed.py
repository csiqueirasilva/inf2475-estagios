from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import numpy as np
import pandas as pd
import torch

from ..data.jobs import TORCH_DEVICE

from ..data.autolabeler import PT_STOPWORDS
from ..data.plot import word_re

import textwrap
from typing import Sequence

OLLAMA_URL = "http://localhost:11434"

from langchain_ollama import ChatOllama

OLLAMA_LLM_LARGER = OllamaLLM(
    model="gemma3:12b",
    base_url = OLLAMA_URL, 
    request_timeout=5,
    temperature=0,
    repeat_penalty=1.5,
    num_predict=1000
)

OLLAMA_LLM = OllamaLLM(
    model="gemma3:12b",
    base_url = OLLAMA_URL, 
    request_timeout=5,
    temperature=0,
    repeat_penalty=1.5,
    num_predict=500
)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NOMIC_MODEL_NAME = "nomic-embed-text"

_MAX_WORDS = 250  # hard budget for the answer

def apply_cv_improvements(
    cv_text: str,
    suggestions: str
) -> str:
    """
    Use the LLM to integrate suggestions into the CV by producing an updated CV text.
    Outputs plain text, preserving original format, without Markdown, special formatting, or placeholders.
    """
    # System prompt: integrate without markdown, no placeholders
    sys_msg = (
        "Você é um assistente que integra recomendações em currículos. "
        "Gere o texto atualizado do currículo em texto simples, sem Markdown ou negrito, "
        "e sem usar placeholders como [Nome], [Data], etc."
    )
    hum_msg = textwrap.dedent(f"""
        CURRÍCULO ORIGINAL:
        {cv_text.strip()}

        SUGESTÕES:
        {suggestions.strip()}

        Instruções:
        1. Incorpore as sugestões diretamente no texto.
        2. Mantenha a formatação original.
        3. Não utilize listas, marcadores, ou numeração automática.
        4. Não adicione placeholders entre colchetes (ex.: [Nome da Instituição], [Data]).
        5. Sua resposta deve conter até {_MAX_WORDS} palavras.
        6. Responda apenas com o texto do currículo atualizado em português brasileiro.
    """)
    response = OLLAMA_LLM_LARGER.invoke([
        ("system", sys_msg),
        ("human", hum_msg)
    ])
    return str(response)


def suggest_cv_improvements(
    cv_text: str,
    job_text: str,
    missing_tokens: Sequence[str],
    *,
    top_k: int = 15,
) -> str:
    """
    Ask the Ollama LLM for short, concrete suggestions so the CV aligns
    better with the Job. Returns plain text with one suggestion per line,
    without Markdown or special formatting.
    """
    sys_msg = (
        "Você é um consultor de carreira. "
        "Liste de 3 a 5 lacunas principais no currículo em relação à vaga. "
        "Responda em texto simples, sem Markdown, listas numeradas ou marcadores."
    )

    hum_msg = textwrap.dedent(f"""\
        CURRÍCULO:
        {cv_text.strip()}

        VAGA:
        {job_text.strip()}

        HABILIDADES/FRASES AUSENTES (top {top_k}):
        {", ".join(missing_tokens[:top_k])}

        Instruções:
        1. Para cada lacuna, descreva uma ação clara em uma linha.
        2. Não ultrapasse {_MAX_WORDS} palavras.
        3. Responda apenas em português brasileiro.
        4. Sem Markdown ou formatação especial.
    """)

    response = OLLAMA_LLM_LARGER.invoke([
        ("system", sys_msg),
        ("human", hum_msg)
    ])
    return str(response)

def get_embed_func(base_url : str = OLLAMA_URL):
    embeddings = OllamaEmbeddings(model=NOMIC_MODEL_NAME, base_url = base_url)
    embed_func = embeddings.embed_documents
    return embed_func
    
class NomicTokenExplainer:
    """
    Build Nomic-space token embeddings once, then retrieve nearest tokens for any 768-d vector.
    """

    def __init__(self, *, batch_size: int = 64):
        self._embed = get_embed_func()
        self.batch_size = batch_size
        self._vocab: list[str] | None = None
        self._vec_vocab: np.ndarray | None = None  # shape (V, 768), row-L2-normalised

    # ......................................................... utilities ...
    @staticmethod
    def _tokens_in(text: str | None) -> list[str]:
        if not text:
            return []
        return [
            w.lower()
            for w in word_re.findall(text)
            if w.lower() not in PT_STOPWORDS and not any(ch.isdigit() for ch in w)
        ]

    def _cache_paths(self):
        return (CACHE_DIR / "token_mat.npy",
                CACHE_DIR / "token_arr.txt")

    def _load_cached(self) -> bool:
        mat_f, txt_f = self._cache_paths()
        if mat_f.exists() and txt_f.exists():
            arr = np.loadtxt(txt_f, dtype=str)
            vec = np.load(mat_f)
            # ensure shapes line up?
            self._vocab = arr.tolist()
            self._vec_vocab = vec
            return True
        return False

    def _maybe_embed_tokens(self, vocab: list[str]) -> np.ndarray:
        """
        Embed `vocab` (unique tokens) via Ollama, batching requests.
        Caches the result to disk for re-use between sessions.
        """
        cache_mat = CACHE_DIR / "token_mat.npy"
        cache_txt = CACHE_DIR / "token_arr.txt"

        # If cache matches, load and return
        if cache_mat.exists() and cache_txt.exists():
            arr_txt = np.loadtxt(cache_txt, dtype=str)
            if len(arr_txt) == len(vocab) and set(arr_txt) == set(vocab):
                vec = np.load(cache_mat)
                return vec

        # Otherwise embed afresh
        vecs: list[list[float]] = []
        for i in range(0, len(vocab), self.batch_size):
            chunk = vocab[i : i + self.batch_size]
            vecs.extend(self._embed(chunk))

        vec = np.asarray(vecs, dtype=np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)  # row-normalise

        # Save cache
        np.save(cache_mat, vec)
        Path(cache_txt).write_text("\n".join(vocab), encoding="utf-8")
        return vec

    def clear_cache(self) -> None:
        for p in (CACHE_DIR / "token_mat.npy", CACHE_DIR / "token_arr.txt"):
            if p.exists():
                p.unlink()
                
    # ......................................................... public API ...
    def build_vocab_from_texts(self, texts: Iterable[str]) -> None:
        """
        Extract tokens from `texts`, embed them (if not cached) and keep them in memory.
        Call once per CLI invocation before using `nearest_tokens()`.
        """
        vocab = sorted({tok for txt in texts for tok in self._tokens_in(txt)},
                       key=str.lower)
        if not vocab:
            raise ValueError("No tokens extracted.")
        self._vocab = vocab
        self._vec_vocab = self._maybe_embed_tokens(vocab)

    def nearest_tokens(self, vec768: np.ndarray, *, k: int = 10) -> list[str]:
        """
        Return the `k` tokens in the current vocab whose embedding is
        closest in cosine similarity to `vec768`.
        """
        # lazy‐load from disk if needed
        if self._vocab is None or self._vec_vocab is None:
            if not self._load_cached():
                raise RuntimeError(
                    "No in-memory vocab and cache not found. "
                    "Run build_vocab_from_texts() first."
                )
        # rest is unchanged
        v = vec768.reshape(-1)
        vn = v / np.linalg.norm(v)
        sims = self._vec_vocab @ vn
        top_idx = sims.argsort()[-k:][::-1]
        return [self._vocab[i] for i in top_idx]
    
class LatentTokenExplainer:
    """
    Encode every token with an AE encoder → cache to disk.
    Caches are kept separate per autoencoder via `suffix`.
    """

    def __init__(
        self,
        ae_encoder: torch.nn.Module,
        *,
        batch_size: int = 128,
        suffix: str = "cv",                 # ✱ new
    ):
        self.enc = ae_encoder.to(TORCH_DEVICE).eval()
        self.batch = batch_size
        self.suffix = suffix.lower()
        self.token_arr: np.ndarray | None = None
        self.token_lat: np.ndarray | None = None  # (V, latent_dim)

    # ------------- helpers ----------------
    def _cache_paths(self) -> tuple[Path, Path]:
        mat = CACHE_DIR / f"token_lat_{self.suffix}.npy"
        txt = CACHE_DIR / f"token_arr_lat_{self.suffix}.txt"
        return mat, txt

    def clear_cache(self) -> None:
        for p in self._cache_paths():
            if p.exists():
                p.unlink()

    # ------------ cache I/O ---------------------------------
    def _load_cached(self) -> bool:
        """Return True if we successfully loaded an existing cache."""
        mat_f, txt_f = self._cache_paths()
        if mat_f.exists() and txt_f.exists():
            self.token_arr = np.loadtxt(txt_f, dtype=str)
            self.token_lat = np.load(mat_f)
            return True
        return False

    # ------------ public ------------------------------------
    def nearest_tokens(self, z: np.ndarray, k: int = 10) -> list[str]:
        # auto-load if nothing in memory
        if self.token_arr is None or self.token_lat is None:
            if not self._load_cached():
                raise RuntimeError(
                    "Latent token cache not found – run "
                    "`internship sanitize latent-token-cache` first."
                )
        z = z / np.linalg.norm(z)
        sims = (self.token_lat @ z) / np.linalg.norm(self.token_lat, axis=1)
        idx  = sims.argsort()[-k:][::-1]
        return self.token_arr[idx].tolist()

    # ------------- main build ------------
    def build_cache(self, tokens: Sequence[str]) -> None:
        toks = sorted(set(tokens))
        mat_f, txt_f = self._cache_paths()

        if mat_f.exists() and txt_f.exists():
            arr_old = np.loadtxt(txt_f, dtype=str)
            if len(arr_old) == len(toks) and set(arr_old) == set(toks):
                self.token_arr = arr_old
                self.token_lat = np.load(mat_f)
                return

        emb_fn = OllamaEmbeddings(model=NOMIC_MODEL_NAME).embed_documents
        lat_chunks: list[np.ndarray] = []
        for i in range(0, len(toks), self.batch):
            chunk = toks[i : i + self.batch]
            with torch.no_grad():
                vec768 = torch.tensor(emb_fn(chunk), dtype=torch.float32,
                                      device=TORCH_DEVICE)
                z = self.enc(vec768)[-1]          # encoder output
            lat_chunks.append(z.cpu().numpy())

        self.token_arr = np.array(toks)
        self.token_lat = np.vstack(lat_chunks)
        np.save(mat_f, self.token_lat)
        txt_f.write_text("\n".join(toks), encoding="utf-8")

# ----------------------------------------------------- tiny helper
def tokens_in(text: str) -> list[str]:
    return [w.lower() for w in word_re.findall(text)
            if w.lower() not in PT_STOPWORDS and not any(ch.isdigit() for ch in w)]
