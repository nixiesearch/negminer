from negminer.log import setup_logging
from negminer.args import parse_args
from negminer.score.ranker import Ranker
from datasets import Dataset
from accelerate import Accelerator
from typing import Dict, List, Any

logger = setup_logging()


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    logger.info("Loaded tokenized query-doc pairs")
    pairs = Dataset.load_from_disk(f"{args.out_dir}/ce_tokenized")

    logger.info("loading qrels")
    qrels = Dataset.load_from_disk(f"{args.out_dir}/qrels_mined")

    accelerator.wait_for_everyone()

    scorer = Ranker(args, accelerator)
    scores = scorer.score(pairs)
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        logger.info("saving")

        def join_scores(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            ce_scores = []
            for qid, docids in zip(batch["query"], batch["docs"]):
                qid_ce_scores = []
                for docid in docids:
                    qid_ce_scores.append(scores[(qid, docid)])
                ce_scores.append(qid_ce_scores)
            batch["doc_ce_scores"] = ce_scores
            return batch

        result = qrels.map(function=join_scores, batched=True, desc="joining")
        print(result.features)
        result.save_to_disk(f"{args.out_dir}/qrels_scored")

    accelerator.end_training()
