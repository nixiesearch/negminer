from negminer.args import parse_args
from negminer.log import setup_logging
from datasets import Dataset
from negminer.search.index import Indexer
from negminer.embed.prepare.data import Qrel
from typing import Dict, List, Any
from tqdm import tqdm

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances
import uuid
import os
import numpy as np

logger = setup_logging()


if __name__ == "__main__":
    args = parse_args()
    queries = Dataset.load_from_disk(f"{args.out_dir}/queries_embed").with_format("np")
    corpus = Dataset.load_from_disk(f"{args.out_dir}/corpus_embed").with_format("np")
    qrels = Dataset.load_from_disk(f"{args.out_dir}/qrels")

    logger.info("Loading embeddings")
    corpus_embed_column = corpus["embed"]
    queries_embed_column = queries["embed"]

    query_ids = queries["id"]
    query_id_to_idx = {id: idx for idx, id in enumerate(query_ids)}

    doc_ids = corpus["id"]
    doc_id_to_idx = {id: idx for idx, id in enumerate(doc_ids)}

    index = Indexer(corpus_embed_column, m=args.hnsw_m, efc=args.hnsw_ef_construction, ef=args.hnsw_ef_search)
    similarities, docs_indexes = index.search(queries_embed_column, args.negatives_candidates_count)

    qrels_dict: Dict[str, List[Qrel]] = dict()
    for qid, docids, scores in tqdm(zip(qrels["query"], qrels["doc"], qrels["score"]), desc="caching qrels"):
        qrels_dict[qid] = [Qrel(docid, score) for docid, score in zip(docids, scores)]

    logger.info(f"Building dataset with search results: {len(query_ids)} queries, {docs_indexes.shape} docs")
    found = Dataset.from_dict({"found_query_id": query_ids})

    def join_queries(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        out_qid: List[str] = []
        out_doc_ids: List[List[str]] = []
        out_doc_scores: List[List[float]] = []
        out_query_doc_cos: List[List[float]] = []
        out_doc_doc_cos: List[List[float]] = []

        for qid in batch["found_query_id"]:
            query_idx = query_id_to_idx[qid]
            found_docnums = docs_indexes[query_idx]
            found_sims = similarities[query_idx]
            if qid in qrels_dict:
                # adding existing qrels first
                qid_rels = qrels_dict[qid]
                qid_qrel_docids = [rel.docid for rel in qid_rels]
                qid_doc_ids = [rel.docid for rel in qid_rels]
                qid_doc_scores = [rel.score for rel in qid_rels]
                qid_doc_cos = [
                    1 - cosine(queries_embed_column[query_idx], corpus_embed_column[doc_id_to_idx[docid]])
                    for docid in qid_doc_ids
                ]

                # then all we found
                found_docids = [doc_ids[num] for num in found_docnums]
                for docid, cos in zip(found_docids, found_sims):
                    if docid not in qid_qrel_docids:
                        qid_doc_ids.append(docid)
                        qid_doc_scores.append(0.0)
                        qid_doc_cos.append(1 - cos)
                # all to all docs
                qid_doc_indexes = [doc_id_to_idx[doc] for doc in qid_doc_ids]
                qid_doc_embeds = corpus_embed_column[qid_doc_indexes]
                doc_doc_dists = 1 - pairwise_distances(qid_doc_embeds, metric="cosine").reshape(-1)

                # add to buffer
                out_qid.append(qid)
                out_doc_ids.append(qid_doc_ids)
                out_doc_scores.append(qid_doc_scores)
                out_query_doc_cos.append(qid_doc_cos)
                out_doc_doc_cos.append(doc_doc_dists)

        return {
            "query": out_qid,
            "docs": out_doc_ids,
            "doc_scores": out_doc_scores,
            "query_doc_cos": out_query_doc_cos,
            "doc_doc_cos": out_doc_doc_cos,
        }

    result = found.map(
        function=join_queries,
        batched=True,
        desc="joining",
        remove_columns=["found_query_id"],
        num_proc=4,
        new_fingerprint=str(
            uuid.uuid4()
        ),  # random one, as otherwise we pickle all the referenced embed dics which takes ages
    )

    result.save_to_disk(f"{args.out_dir}/qrels_mined")
