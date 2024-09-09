from negminer.log import setup_logging
from negminer.args import parse_args
from datasets import Dataset
import os
from typing import Dict, List, Any
from transformers import AutoTokenizer

logger = setup_logging()

if __name__ == "__main__":
    args = parse_args()
    logger.info("Loaded cached qrels_mined dataset")
    qrels = Dataset.load_from_disk(f"{args.out_dir}/qrels_mined")

    logger.info("Loading corpus in RAM")
    corpus_dict = Dataset.load_from_disk(f"{args.out_dir}/corpus").select_columns(["id", "doc"]).to_dict()
    corpus = {id: doc for id, doc in zip(corpus_dict["id"], corpus_dict["doc"])}

    logger.info("Loading queries in RAM")
    queries_dict = Dataset.load_from_disk(f"{args.out_dir}/queries").select_columns(["id", "doc"]).to_dict()
    queries = {id: doc for id, doc in zip(queries_dict["id"], queries_dict["doc"])}
    tokenizer = AutoTokenizer.from_pretrained(args.ce)

    def tokenize(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        pairs = []
        pair_qids = []
        pair_docids = []
        for qid, docids in zip(batch["query"], batch["docs"]):
            for docid in docids:
                if qid in queries and docid in corpus:
                    pair_qids.append(qid)
                    pair_docids.append(docid)
                    pairs.append([queries[qid], corpus[docid]])
                else:
                    logger.warn(
                        f"query {qid}={queries.get(qid)} or doc {docid}={corpus.get(docid)} missing in dataset"
                    )

        tokenized = tokenizer(pairs, padding=False, truncation=True, max_length=args.ce_seq_len)
        input_ids = [iid for iid in tokenized["input_ids"]]
        length = [len(iid) for iid in tokenized["input_ids"]]
        return {"tokenized": input_ids, "length": length, "docid": pair_docids, "qid": pair_qids}

    tokenized = qrels.map(
        function=tokenize,
        batched=True,
        desc="tokenizing",
        remove_columns=qrels.column_names,
        num_proc=os.cpu_count(),
    )
    tokenized = tokenized.sort(column_names=["length"], reverse=True)
    tokenized = tokenized.flatten_indices(num_proc=os.cpu_count())
    tokenized.save_to_disk(f"{args.out_dir}/ce_tokenized")
