"""
FAISS vector index with persistent storage and ID mapping.

Uses IndexFlatIP (inner product) on L2-normalized vectors, which is
equivalent to cosine similarity search.

Includes a NumPy-based fallback if 'faiss' is not installed (e.g. on Python 3.14 on Windows).
"""
import os
import json
import numpy as np
from typing import List, Tuple, Optional
import pickle

from embeddings import EMBEDDING_DIM

# ── Dynamic Import / Fallback ──────────────────────────────────────────────

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: 'faiss' module not found. Using NumPy-based fallback index.")

# ── NumPy Fallback Implementation ──────────────────────────────────────────

class NumpyFlatIndex:
    """A minimal in-memory implementation of faiss.IndexFlatIP using NumPy."""
    def __init__(self, d: int):
        self.d = d
        # Shape: (N, d)
        self.vectors = np.zeros((0, d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]

    def add(self, x: np.ndarray):
        """Add vectors of shape (n, d) or (d,)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.d:
            raise ValueError(f"Expected dim {self.d}, got {x.shape[1]}")
        self.vectors = np.vstack([self.vectors, x])

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for x (shape (1, d)) in the index.
        Returns (distances, indices) arrays of shape (1, k).
        """
        if self.ntotal == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Inner product: (1, d) @ (d, N) -> (1, N)
        scores = np.dot(x, self.vectors.T) 
        
        # Get top-k indices
        k = min(k, self.ntotal)
        indices = np.argsort(scores[0])[-k:][::-1]
        top_scores = scores[0][indices]

        return top_scores.reshape(1, -1), indices.reshape(1, -1)

    def save(self, path: str):
        """Save vectors to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.vectors, f)

    def load(self, path: str):
        """Load vectors from disk."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.vectors = pickle.load(f)

# ── Main Wrapper ───────────────────────────────────────────────────────────

class FaissIndex:
    """Wrapper around a FAISS (or NumPy) flat inner-product index with ID mapping."""

    def __init__(self, index_path: str, mapping_path: str):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: dict[int, str] = {}
        self._next_idx = 0
        self.is_numpy = not HAS_FAISS

        if HAS_FAISS:
            if os.path.exists(index_path):
                try:
                    self.index = faiss.read_index(index_path)
                    print(f"FAISS index loaded: {self.index.ntotal} vectors")
                except:
                    self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            else:
                self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        else:
            self.index = NumpyFlatIndex(EMBEDDING_DIM)
            if os.path.exists(index_path):
                try: self.index.load(index_path)
                except: pass

        self._load_mapping()
        self._next_idx = self.index.ntotal

    def add(self, embedding: np.ndarray, content_id: str):
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        idx = self._next_idx
        self.id_to_idx[content_id] = idx
        self.idx_to_id[idx] = content_id
        self._next_idx += 1
        self.save()

    def search(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0: return []
        k = min(k, self.index.ntotal)
        vec = embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(vec, k)
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            if idx >= 0 and idx in self.idx_to_id:
                results.append((self.idx_to_id[idx], score))
        return results

    def remove(self, content_id: str) -> bool:
        if content_id not in self.id_to_idx: return False
        
        # Simple rebuild logic for small datasets
        n = self.index.ntotal
        if n <= 1:
            if self.is_numpy: self.index = NumpyFlatIndex(EMBEDDING_DIM)
            else: self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.id_to_idx.clear(); self.idx_to_id.clear(); self._next_idx = 0
            self.save()
            return True

        # Reconstruct and filter
        # (Assuming small scale for WebAR POC)
        # Note: In production, you'd pull from DB to rebuild index
        return False # Placeholder for complex rebuild

    @property
    def total(self) -> int: return self.index.ntotal

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if self.is_numpy: self.index.save(self.index_path)
        else: faiss.write_index(self.index, self.index_path)
        mapping_data = {
            "id_to_idx": self.id_to_idx,
            "idx_to_id": {str(k): v for k, v in self.idx_to_id.items()},
            "next_idx": self._next_idx,
        }
        with open(self.mapping_path, "w") as f:
            json.dump(mapping_data, f, indent=2)

    def _load_mapping(self):
        if not os.path.exists(self.mapping_path): return
        with open(self.mapping_path, "r") as f:
            data = json.load(f)
        self.id_to_idx = data.get("id_to_idx", {})
        self.idx_to_id = {int(k): v for k, v in data.get("idx_to_id", {}).items()}
        self._next_idx = data.get("next_idx", 0)
