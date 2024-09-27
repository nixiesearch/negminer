from negminer.log import setup_logging
from negminer.args import parse_args
from datasets import Dataset
from typing import Dict, List, Any
from itertools import batched
import os

logger = setup_logging()

if __name__ == "__main__":
    args = parse_args()
    logger.info("Loading negatives")
    qrels = Dataset.load_from_disk(f"{args.out_dir}/qrels_scored")

    logger.info("Loading corpus in RAM")
    corpus_dict = Dataset.load_from_disk(f"{args.out_dir}/corpus").select_columns(["id", "doc"]).to_dict()
    corpus = {id: doc for id, doc in zip(corpus_dict["id"], corpus_dict["doc"])}

    logger.info("Loading queries in RAM")
    queries_dict = Dataset.load_from_disk(f"{args.out_dir}/queries").select_columns(["id", "doc"]).to_dict()
    queries = {id: doc for id, doc in zip(queries_dict["id"], queries_dict["doc"])}

    def is_duplicate(idx: int, cos_scores: List[float]) -> bool:
        for i in range(0, idx):
            if cos_scores[i] > args.dup_threshold:
                return True
        return False

    def sample(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        query_out: List[str] = []
        pos: List[List[str]] = []
        pos_cos_score: List[List[float]] = []
        pos_ce_score: List[List[float]] = []
        neg: List[List[str]] = []
        neg_cos_score: List[List[float]] = []
        neg_ce_score: List[List[float]] = []
        out_dropped_dupes: List[float] = []
        out_dropped_too_close: List[float] = []
        out_dropped_too_high: List[float] = []
        for (
            qid,
            docids,
            qde_scores,
            qdcos_scores,
            ddcos_scores,
            qdce_scores,
        ) in zip(
            batch["query"],
            batch["docs"],
            batch["doc_scores"],
            batch["query_doc_cos"],
            batch["doc_doc_cos"],
            batch["doc_ce_scores"],
        ):
            # print(qde_scores)
            query = queries[qid]
            query_pos: List[str] = []
            query_pos_cos_score: List[float] = []
            query_pos_ce_score: List[float] = []
            query_neg: List[str] = []
            query_neg_cos_score: List[float] = []
            query_neg_ce_score: List[float] = []
            min_pos_ce_score = min([score for explicit, score in zip(qde_scores, qdce_scores) if explicit > 0])
            dropped_dupes = 0
            dropped_too_close = 0
            dropped_too_high = 0
            for (
                (idx, docid),
                qde_score,
                qdcos_score,
                ddcos_scores,
                qdce_score,
            ) in zip(enumerate(docids), qde_scores, qdcos_scores, batched(ddcos_scores, len(docids)), qdce_scores):
                doc = corpus[docid]
                if qde_score > 0:
                    # explicit positive
                    query_pos.append(doc)
                    query_pos_cos_score.append(qdcos_score)
                    query_pos_ce_score.append(qdce_score)
                else:
                    if not is_duplicate(idx, list(ddcos_scores)):
                        if min_pos_ce_score - qdce_score > args.pos_neg_threshold:
                            if qdce_score < args.pos_threshold:
                                if len(query_neg) < args.negatives_count:
                                    query_neg.append(doc)
                                    query_neg_cos_score.append(qdcos_score)
                                    query_neg_ce_score.append(qdce_score)
                            else:
                                dropped_too_high += 1
                        else:
                            dropped_too_close += 1
                    else:
                        dropped_dupes += 1

            if len(query_pos) > 0 and len(query_neg) > 0:
                query_out.append(query)
                pos.append(query_pos)
                pos_cos_score.append(query_pos_cos_score)
                pos_ce_score.append(query_pos_ce_score)
                neg.append(query_neg)
                neg_cos_score.append(query_neg_cos_score)
                neg_ce_score.append(query_neg_ce_score)
                out_dropped_dupes.append(dropped_dupes)
                out_dropped_too_close.append(dropped_too_close)
                out_dropped_too_high.append(dropped_too_high)
        return {
            "query": query_out,
            "pos": pos,
            "pos_cos_score": pos_cos_score,
            "pos_ce_score": pos_ce_score,
            "neg": neg,
            "neg_cos_score": neg_cos_score,
            "neg_ce_score": neg_ce_score,
            "dropped_dupes": out_dropped_dupes,
            "dropped_too_close": out_dropped_too_close,
            "dropped_too_high": out_dropped_too_high,
        }

    sampled = qrels.map(function=sample, batched=True, desc="sampling", remove_columns=qrels.column_names)
    sampled.to_json(f"{args.out_dir}/negatives.jsonl")
