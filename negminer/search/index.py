import numpy as np
from typing import Tuple
from dataclasses import dataclass
import faiss
import logging
from tqdm import tqdm
import os

logger = logging.getLogger()


@dataclass
class DocScore:
    doc: str
    score: float


class Indexer:
    # def __init__(self, docs: np.ndarray, m: int = 64, efc: int = 256, ef: int = 128):
    def __init__(self, docs: np.ndarray, m: int = 16, efc: int = 32, ef: int = 24):
        logger.info("HNSW index build started")
        dims = docs.shape[1]
        self.index = faiss.IndexHNSWFlat(dims, m)
        self.index.hnsw.efSearch = ef
        self.index.hnsw.efConstruction = efc
        faiss.omp_set_num_threads(os.cpu_count())
        for chunk in tqdm(np.array_split(docs, 500), desc="indexing"):
            self.index.add(chunk)
        logger.info("HNSW index built successfully")

    def search(self, queries: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        sims = []
        docs = []
        for chunk in tqdm(np.array_split(queries, 100), desc="searching"):
            chunk_sims, chunk_docs = self.index.search(chunk, top_n)
            sims.append(chunk_sims)
            docs.append(chunk_docs)
        return (np.vstack(sims), np.vstack(docs))
