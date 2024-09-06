from dataclasses import dataclass, field
from typing import List, Dict
from datasets import load_dataset
from functools import partial
from transformers import PreTrainedTokenizerBase
from datasets import Dataset
from collections import defaultdict
import os
import logging

logger = logging.getLogger()


@dataclass
class Qrel:
    docid: str
    score: float


class Queries:
    @staticmethod
    def load_corpus(tokenizer: PreTrainedTokenizerBase, path: str, prompt: str, max_len: int) -> Dataset:
        return Queries._load_json(
            tokenizer=tokenizer,
            path=path,
            prompt=prompt,
            need_title=True,
            max_len=max_len,
        )

    @staticmethod
    def load_queries(tokenizer: PreTrainedTokenizerBase, path: str, prompt: str, max_len: int) -> Dataset:
        return Queries._load_json(
            tokenizer=tokenizer,
            path=path,
            prompt=prompt,
            need_title=False,
            max_len=max_len,
        )

    @staticmethod
    def _load_json(
        tokenizer: PreTrainedTokenizerBase, path: str, prompt: str, need_title: bool, max_len: int
    ) -> Dataset:
        def maybe_join(title: str, text: str) -> str:
            if len(title) > 0:
                return title + " " + text
            else:
                return text

        def join_and_tokenize(prefix: str, need_title: bool, batch: Dict[str, List]) -> Dict[str, List]:
            ids = batch["_id"]
            if need_title:
                docs = [maybe_join(title, text) for title, text in zip(batch["title"], batch["text"])]
            else:
                docs = batch["text"]
            docs_formatted = [prefix + doc for doc in docs]
            docs_tokenized_enc = tokenizer(
                docs_formatted, padding=False, truncation=True, max_length=max_len, return_tensors="np"
            )
            docs_tokenized = [doc for doc in docs_tokenized_enc["input_ids"]]
            length = [len(doc) for doc in docs_tokenized_enc["input_ids"]]
            return {"id": ids, "doc": docs, "tokenized": docs_tokenized, "length": length}

        num_cpus = os.cpu_count()
        dataset = load_dataset("json", data_files=path, split="train")
        fields_to_drop = ["_id", "text"]
        if need_title:
            fields_to_drop.append("title")
        dataset = dataset.map(
            function=partial(join_and_tokenize, prompt, need_title),
            batched=True,
            num_proc=num_cpus,
            remove_columns=fields_to_drop,
        )
        dataset = dataset.sort(column_names="length", reverse=True)
        dataset = dataset.flatten_indices(num_proc=num_cpus)
        return dataset

    @staticmethod
    def load_qrels(path: str, corpus: Dataset, queries: Dataset) -> Dataset:
        doc_ids = set(corpus["id"])
        query_ids = set(queries["id"])
        qrels = load_dataset("csv", data_files=path, split="train", delimiter="\t").to_dict()
        qreldict: Dict[str, List[Qrel]] = defaultdict(list)
        mismatch_count = 0
        for qid, docid, score in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"]):
            if qid in query_ids and docid in doc_ids:
                qreldict[qid].append(Qrel(docid, float(score)))
            else:
                mismatch_count += 1
        qrel_qids = []
        qrel_docids = []
        qrel_scores = []
        for qid, qrs in qreldict.items():
            qrel_qids.append(qid)
            qrel_docids.append([qrel.docid for qrel in qrs])
            qrel_scores.append([qrel.score for qrel in qrs])
        qrels_dataset = Dataset.from_dict({"query": qrel_qids, "doc": qrel_docids, "score": qrel_scores})
        logger.info(f"Loaded {len(qrel_qids)} queries, dropped {mismatch_count} broken refs")
        return qrels_dataset
