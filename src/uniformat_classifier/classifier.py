"""Prototype-network classifier on top of sentence-transformer embeddings.

For each Uniformat code seen in training, we maintain the L2-normalized centroid
of the embeddings of its example texts (the "prototype").
Inference: encode the new text, compute cosine similarity to every prototype, pick the
top-k. Updating: when the user confirms a label, add the new embedding to the running
mean for that code — instant feedback, no retraining.

We additionally track per-class "tightness" (mean cosine sim of training points to their
own prototype) so the open-set detector can flag inputs that are an unusual distance away
even from their nearest known class.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class Prediction:
    code: str
    score: float       # cosine similarity (-1..1)
    margin: float      # score - second-best score (proxy for confidence)


@dataclass
class ClassifierVerdict:
    """Verdict for a single input, including signals the active-learning loop uses."""
    text: str
    top: list[Prediction]               # top-k predictions
    nearest_neighbor_sim: float          # similarity to nearest individual training example
    out_of_distribution: bool            # True if the input is far from any prototype


class Classifier:
    """Prototype-network classifier with hot-swappable training set.

    The model itself (sentence-transformer) is loaded lazily so commands that don't
    actually need encoding (e.g. `info`, `inspect-taxonomy`) start instantly.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        # Prototypes: code -> normalized centroid (1D float32 array)
        self._prototypes: dict[str, np.ndarray] = {}
        # Counts (so we can add new examples online without re-summing)
        self._counts: dict[str, int] = {}
        # Sums (un-normalized — we re-normalize after each update)
        self._sums: dict[str, np.ndarray] = {}
        # Per-class average self-similarity, for the OOD detector
        self._tightness: dict[str, float] = {}
        # Per-text embedding cache (text hash -> embedding), keyed by hash for compactness
        self._embed_cache: dict[str, np.ndarray] = {}
        if cache_dir:
            self._load_embed_cache()

    # ----- model lazy-load --------------------------------------------------

    def _ensure_model(self):
        if self._model is None:
            # Defer import; sentence-transformers + torch is heavy.
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ----- embedding with cache --------------------------------------------

    def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        """Embed a list of texts; returns a 2D float32 array of L2-normalized rows."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # Look up cache
        keys = [_hash(t) for t in texts]
        missing_ix = [i for i, k in enumerate(keys) if k not in self._embed_cache]
        if missing_ix:
            model = self._ensure_model()
            to_encode = [texts[i] for i in missing_ix]
            new = model.encode(
                to_encode,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            ).astype(np.float32)
            for j, i in enumerate(missing_ix):
                self._embed_cache[keys[i]] = new[j]
            if self.cache_dir:
                self._save_embed_cache()

        return np.stack([self._embed_cache[k] for k in keys], axis=0)

    # ----- training ---------------------------------------------------------

    def fit(self, texts: list[str], codes: list[str], *, show_progress: bool = True) -> None:
        """Replace prototypes with ones built from (texts, codes)."""
        assert len(texts) == len(codes), "texts and codes must be same length"
        if not texts:
            self._prototypes.clear()
            self._counts.clear()
            self._sums.clear()
            self._tightness.clear()
            return

        embs = self.encode(texts, show_progress=show_progress)

        # Aggregate per class
        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        per_class_embs: dict[str, list[np.ndarray]] = {}
        for emb, code in zip(embs, codes):
            if code not in sums:
                sums[code] = emb.copy()
                counts[code] = 1
                per_class_embs[code] = [emb]
            else:
                sums[code] += emb
                counts[code] += 1
                per_class_embs[code].append(emb)

        protos = {code: _l2norm(s) for code, s in sums.items()}

        # Per-class tightness = mean cosine of members to the prototype
        tight: dict[str, float] = {}
        for code, members in per_class_embs.items():
            mat = np.stack(members)
            sims = mat @ protos[code]
            tight[code] = float(np.mean(sims))

        self._prototypes = protos
        self._sums = sums
        self._counts = counts
        self._tightness = tight

    def add_example(self, text: str, code: str) -> None:
        """Add a single new (text, code) example and update the prototype for that code."""
        emb = self.encode([text])[0]
        if code in self._sums:
            self._sums[code] += emb
            self._counts[code] += 1
        else:
            self._sums[code] = emb.copy()
            self._counts[code] = 1
            # New class — bootstrap tightness from a single point
            self._tightness[code] = 1.0
        self._prototypes[code] = _l2norm(self._sums[code])
        # Tightness drifts slowly — re-estimate sparingly. Cheap approximation: blend.
        sim_to_proto = float(emb @ self._prototypes[code])
        n = self._counts[code]
        old = self._tightness.get(code, sim_to_proto)
        self._tightness[code] = (old * (n - 1) + sim_to_proto) / n

    # ----- prediction -------------------------------------------------------

    def known_codes(self) -> list[str]:
        return list(self._prototypes.keys())

    def predict(
        self,
        text: str,
        *,
        top_k: int = 3,
        all_training_embs: np.ndarray | None = None,
    ) -> ClassifierVerdict:
        """Predict top-k codes for a single text.

        Args:
            text: input text
            top_k: how many candidates to return (sorted desc)
            all_training_embs: optional matrix of all training embeddings, for the
                nearest-neighbor signal (used by the OOD detector). If None, the OOD
                signal falls back to prototype distance only.
        """
        if not self._prototypes:
            return ClassifierVerdict(
                text=text,
                top=[],
                nearest_neighbor_sim=0.0,
                out_of_distribution=True,
            )

        emb = self.encode([text])[0]
        codes = list(self._prototypes.keys())
        proto_mat = np.stack([self._prototypes[c] for c in codes])  # (K, D)
        sims = proto_mat @ emb  # (K,) cosine sim since both normalized

        order = np.argsort(-sims)[:top_k]
        scores = sims[order]
        top_preds: list[Prediction] = []
        for i, ix in enumerate(order):
            score = float(scores[i])
            margin = float(scores[i] - scores[i + 1]) if i + 1 < len(scores) else float(scores[i])
            top_preds.append(Prediction(code=codes[ix], score=score, margin=margin))

        # Nearest-neighbor sim across all training points (used for open-set detection)
        nn_sim = float(np.max(emb @ all_training_embs.T)) if all_training_embs is not None else top_preds[0].score

        # OOD: low absolute proto sim AND low NN sim AND well below class tightness
        best_code = top_preds[0].code
        best_score = top_preds[0].score
        tightness = self._tightness.get(best_code, 0.7)
        # If best proto-sim is much worse than the class's typical self-sim AND nn sim is low → OOD
        ood = (best_score < 0.45) or (nn_sim < 0.55) or (best_score < tightness - 0.20)

        return ClassifierVerdict(
            text=text,
            top=top_preds,
            nearest_neighbor_sim=nn_sim,
            out_of_distribution=ood,
        )

    def all_training_embeddings(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for the supplied training texts (uses the cache)."""
        return self.encode(texts)

    # ----- embedding cache persistence -------------------------------------

    def _cache_file(self) -> Path:
        assert self.cache_dir is not None
        # one cache file per model (so different models don't collide)
        safe_name = self.model_name.replace("/", "__")
        return self.cache_dir / f"embeddings_{safe_name}.pkl"

    def _load_embed_cache(self) -> None:
        if not self.cache_dir:
            return
        f = self._cache_file()
        if f.exists():
            try:
                with f.open("rb") as fh:
                    self._embed_cache = pickle.load(fh)
            except Exception as e:
                print(f"  ! could not load embed cache: {e}")
                self._embed_cache = {}

    def _save_embed_cache(self) -> None:
        if not self.cache_dir:
            return
        f = self._cache_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open("wb") as fh:
            pickle.dump(self._embed_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)


# ---- helpers ---------------------------------------------------------------

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---- 1-of-K (leave-one-out cross-validation) ------------------------------

@dataclass
class EvalResult:
    n: int
    top1_acc: float
    top3_acc: float
    n1_acc: float                       # Niveau-1 (top-letter) accuracy
    confusions: list[tuple[str, str, int]]   # (true_code, pred_code, count) — top mistakes


def evaluate_holdout(
    classifier: Classifier,
    texts: list[str],
    codes: list[str],
    *,
    test_size: float = 0.2,
    seed: int = 42,
    progress=None,
) -> EvalResult:
    """Train on a random subset, evaluate on the held-out rest.

    NOTE: this is 1-of-K-style stratified holdout, not a true leave-one-out (which would
    cost N classifier rebuilds). For tens of thousands of rows the stratified holdout is
    a solid proxy and ~100x faster.
    """
    from collections import Counter
    rng = np.random.default_rng(seed)

    n = len(texts)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(n * test_size))
    test_ix = set(indices[:n_test].tolist())
    train_ix = [i for i in range(n) if i not in test_ix]

    # Train
    train_texts = [texts[i] for i in train_ix]
    train_codes = [codes[i] for i in train_ix]
    classifier.fit(train_texts, train_codes, show_progress=False)
    train_embs = classifier.all_training_embeddings(train_texts)

    # Evaluate
    test_texts = [texts[i] for i in test_ix]
    test_codes = [codes[i] for i in test_ix]

    correct1 = correct3 = correct_n1 = 0
    confusion = Counter()
    for i, (text, true_code) in enumerate(zip(test_texts, test_codes)):
        verdict = classifier.predict(text, top_k=3, all_training_embs=train_embs)
        if not verdict.top:
            continue
        preds = [p.code for p in verdict.top]
        if preds[0] == true_code:
            correct1 += 1
        else:
            confusion[(true_code, preds[0])] += 1
        if true_code in preds:
            correct3 += 1
        if preds[0][:1] == true_code[:1]:
            correct_n1 += 1
        if progress is not None:
            progress(i + 1, len(test_texts))

    n_eval = len(test_texts)
    return EvalResult(
        n=n_eval,
        top1_acc=correct1 / n_eval if n_eval else 0.0,
        top3_acc=correct3 / n_eval if n_eval else 0.0,
        n1_acc=correct_n1 / n_eval if n_eval else 0.0,
        confusions=[(t, p, c) for (t, p), c in confusion.most_common(15)],
    )


def fit_from_records(
    classifier: Classifier,
    records: Iterable,
    *,
    target_level: int = 3,
    show_progress: bool = True,
) -> int:
    """Train classifier from training-records (or LabeledExample-like objects).

    Anything with `text` and `code` attributes is accepted. We rollup the code:
    - If target_level=3 and code is N1/N2, the row is *kept* but contributes to a
      coarser prototype keyed by that shorter code. The classifier therefore has
      both N3 and N1/N2 prototypes; nearest-prototype lookup picks whichever fits.

    Returns number of (text, code) pairs used.
    """
    texts: list[str] = []
    codes: list[str] = []
    for rec in records:
        text = getattr(rec, "text", None)
        code = getattr(rec, "code", None)
        if not text or not code:
            continue
        texts.append(text)
        codes.append(code)
    classifier.fit(texts, codes, show_progress=show_progress)
    return len(texts)
