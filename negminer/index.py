import numpy as np
from typing import Tuple
from dataclasses import dataclass
import faiss
import logging

logger = logging.getLogger()


@dataclass
class DocScore:
    doc: str
    score: float


class Indexer:
    def __init__(self, docs: np.ndarray, m: int = 64, efc: int = 256, ef: int = 128):
        logger.info("HNSW index build started")
        dims = docs.shape[1]
        self.index = faiss.IndexHNSWFlat(dims, m)
        self.index.hnsw.efConstruction = efc
        self.index.hnsw.efSearch = ef
        self.index.add(docs)
        logger.info("HNSW index built successfully")

    def search(self, queries: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.search(queries, top_n)
