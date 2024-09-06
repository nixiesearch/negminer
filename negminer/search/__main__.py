from negminer.args import parse_args
from negminer.log import setup_logging
from datasets import Dataset
from negminer.index import Indexer
from negminer.prepare.data import Qrel
from typing import Dict, List
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine

logger = setup_logging()


if __name__ == "__main__":
    args = parse_args()
    queries = Dataset.load_from_disk(f"{args.out_dir}/queries_embed").with_format("np")
    corpus = Dataset.load_from_disk(f"{args.out_dir}/corpus_embed").with_format("np")
    qrels = Dataset.load_from_disk(f"{args.out_dir}/qrels")

    logger.info("Loading embeddings")
    corpus_embed_col = corpus["embed"]
    queries_embed_col = queries["embed"]

    index = Indexer(corpus_embed_col)
    sims, docs = index.search(queries_embed_col, 16)

    qrels_dict: Dict[str, List[Qrel]] = dict()
    for qid, docids, scores in tqdm(zip(qrels["query"], qrels["doc"], qrels["score"]), desc="caching qrels"):
        qrels_dict[qid] = [Qrel(docid, score) for docid, score in zip(docids, scores)]

    query_ids = queries["id"]
    query_embs = {id: emb for id, emb in tqdm(zip(query_ids, queries_embed_col), desc="caching query embs")}
    doc_ids = corpus["id"]
    doc_embs = {id: emb for id, emb in tqdm(zip(doc_ids, corpus_embed_col), desc="caching doc embs")}

    out_qid: List[str] = []
    out_doc_ids: List[List[str]] = []
    out_doc_scores: List[List[float]] = []
    out_query_doc_cos: List[List[float]] = []
    out_doc_doc_cos: List[List[float]] = []

    qid_no_qrel_count = 0
    mined_negs = 0
    for qid, found_docnums, found_sims in tqdm(zip(query_ids, docs, sims), desc="joining"):
        if qid in qrels_dict:
            # adding existing qrels first
            qid_rels = qrels_dict[qid]
            qid_qrel_docids = [rel.docid for rel in qid_rels]
            qid_doc_ids = [rel.docid for rel in qid_rels]
            qid_doc_scores = [rel.score for rel in qid_rels]
            qid_doc_cos = [1 - cosine(query_embs[qid], doc_embs[docid]) for docid in qid_doc_ids]

            # then all we found
            found_docids = [doc_ids[num] for num in np.flip(found_docnums)]
            for docid, cos in zip(found_docids, np.flip(found_sims)):
                if docid not in qid_qrel_docids:
                    mined_negs += 1
                    qid_doc_ids.append(docid)
                    qid_doc_scores.append(0.0)
                    qid_doc_cos.append(cos)
            # all to all docs
            qid_doc_doc_cos: List[float] = []
            for doc1 in qid_doc_ids:
                for doc2 in qid_doc_ids:
                    qid_doc_doc_cos.append(1 - cosine(doc_embs[doc1], doc_embs[doc2]))
            # add to buffer
            out_qid.append(qid)
            out_doc_ids.append(qid_doc_ids)
            out_doc_scores.append(qid_doc_scores)
            out_query_doc_cos.append(qid_doc_cos)
            out_doc_doc_cos.append(qid_doc_doc_cos)
        else:
            qid_no_qrel_count += 1

    result = Dataset.from_dict(
        {
            "query": out_qid,
            "docs": out_doc_ids,
            "doc_scores": out_doc_scores,
            "query_doc_cos": out_query_doc_cos,
            "doc_doc_cos": out_doc_doc_cos,
        }
    )
    result.save_to_disk(f"{args.out_dir}/qrels_mined")
    logger.info(f"Done: {mined_negs} negatives mined, dropped {qid_no_qrel_count} orphan queries")
