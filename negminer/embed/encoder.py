from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from sentence_transformers.models import Transformer, Pooling
from accelerate import Accelerator
from negminer.args import NegMineArgs
from transformers import AutoTokenizer, BatchEncoding
from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logger = logging.getLogger()


@dataclass
class EncodedStrings:
    keys: List[str]
    embeddings: torch.Tensor


class TextEncoder:
    def __init__(self, args: NegMineArgs, acc: Accelerator) -> None:
        transformer = Transformer(
            model_name_or_path=args.embedding,
            model_args={"torch_dtype": torch.bfloat16},
        )
        pool = Pooling(transformer.get_word_embedding_dimension(), "mean")
        self.model = acc.prepare_model(torch.nn.Sequential(transformer, pool))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.embedding)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.acc = acc
        self.args = args

    def encode(self, dataset: Dataset) -> np.ndarray:
        def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # print(f"lens: {[row["length"] for row in batch]}")
            tokenized = [row["tokenized"] for row in batch]
            encoded = BatchEncoding(
                {
                    "input_ids": tokenized,
                    "attention_mask": [np.full(len(vec), 1) for vec in tokenized],
                }
            )
            padded = self.tokenizer.pad(encoded, padding="longest", pad_to_multiple_of=8, return_tensors="pt")
            return padded

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.embedding_batch_size,
            collate_fn=collate,
            num_workers=4,
            prefetch_factor=2,
            # pin_memory=True,
        )
        acc_loader = self.acc.prepare_data_loader(loader)

        embeddings = []
        progress = tqdm(disable=(not self.acc.is_local_main_process), total=len(loader), desc="embedding")

        for batch in acc_loader:
            with torch.no_grad():
                result: Dict[str, torch.Tensor] = self.model.forward(batch)
                batch_embeddings = result["sentence_embedding"].detach().float()
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                batch_embeddings = self.acc.gather_for_metrics(batch_embeddings).cpu()
                embeddings.append(batch_embeddings)
                progress.update(self.acc.num_processes)
        logger.info("Embedding done, merging results")
        result = torch.vstack(embeddings).numpy()
        logger.info("Merged")
        return result
