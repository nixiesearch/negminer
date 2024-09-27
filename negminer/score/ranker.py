from accelerate import Accelerator
from negminer.args import NegMineArgs
from transformers import AutoModelForSequenceClassification, BatchEncoding, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, List, Any, Union
import logging

logger = logging.getLogger()


class Ranker:
    def __init__(self, args: NegMineArgs, acc: Accelerator) -> None:
        base = AutoModelForSequenceClassification.from_pretrained(
            args.ce,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.ce_attn_implementation,
            trust_remote_code=True,
            load_in_8bit=True,
            # device_map="auto"
        )
        base.config.pad_token_id = 0
        self.model = acc.prepare_model(base)
        self.model.eval()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.ce)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.tokenizer.pad_token = "<unk>"
        self.acc = acc

    def score(self, pairs: Dataset) -> Dict[Tuple[str, str], float]:
        def collate(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.tensor, List[str]]]:
            tokenized = [row["tokenized"] for row in batch]
            encoded = BatchEncoding(
                {
                    "input_ids": tokenized,
                    "attention_mask": [[1] * len(vec) for vec in tokenized],
                }
            )
            padded = self.tokenizer.pad(encoded, padding="longest", pad_to_multiple_of=8, return_tensors="pt")
            return {
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
                "qid": [row["qid"] for row in batch],
                "docid": [row["docid"] for row in batch],
            }

        loader = DataLoader(
            dataset=pairs,
            batch_size=self.args.ce_batch_size,
            collate_fn=collate,
            num_workers=4,
            prefetch_factor=2,
            # pin_memory=True,
        )
        acc_loader = self.acc.prepare_data_loader(loader)

        scores: Dict[Tuple[str, str], float] = dict()
        progress = tqdm(disable=(not self.acc.is_local_main_process), total=len(loader), desc="scoring")

        for batch in acc_loader:
            with torch.no_grad():
                node_result: SequenceClassifierOutput = self.model.forward(batch["input_ids"], batch["attention_mask"])
                result = self.acc.gather_for_metrics(node_result.logits)
                qids = self.acc.gather_for_metrics(batch["qid"])
                docids = self.acc.gather_for_metrics(batch["docid"])
                for qid, docid, score in zip(qids, docids, result.reshape(1, -1).tolist()[0]):
                    scores[(qid, docid)] = score
                progress.update(self.acc.num_processes)
        logger.info("Scoring done")
        return scores
