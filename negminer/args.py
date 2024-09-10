from dataclasses import dataclass, field
from transformers import HfArgumentParser
import sys
import yaml


@dataclass
class NegMineArgs:
    corpus: str = field(metadata={"help": "path to corpus.json"})
    queries: str = field(metadata={"help": "path to queries.json"})
    qrels: str = field(metadata={"help": "path to qrels.tsv"})
    embedding: str = field(metadata={"help": "name or path for the embedding model"})
    ce: str = field(metadata={"help": "name or path for the ranking model"})
    out_dir: str = field(metadata={"help": "path to write stuff to"})
    embedding_seq_len: int = field(default=512, metadata={"help": "max seq len in tokens for the embedding model"})
    embedding_batch_size: int = field(default=32, metadata={"help": "batch size for the embedding model"})
    ce_seq_len: int = field(default=512, metadata={"help": "max seq len in tokens for the ranking model"})
    ce_attn_implementation: str = field(default="eager", metadata={"help": "attn impl"})
    ce_batch_size: int = field(default=32, metadata={"help": "batch size for CE model"})
    doc_prompt: str = field(default="", metadata={"help": "doc prompt"})
    query_prompt: str = field(default="", metadata={"help": "query prompt"})
    negatives_count: int = field(default=8, metadata={"help": "how many negatives to mine"})
    negatives_candidates_count: int = field(
        default=16, metadata={"help": "how many negative candidates to search for"}
    )
    dup_threshold: float = field(
        default=0.95, metadata={"help": "cosine distance between docs to be considered duplicates and dropped"}
    )
    pos_neg_threshold: float = field(
        default=0.05,
        metadata={
            "help": "minimal CE score diff between query-pos and query-neg. negatives with less than this threshold are considered false negatives and dropped"
        },
    )
    hnsw_m: int = field(default=64, metadata={"help": "FAISS HNSW M value"})
    hnsw_ef_search: int = field(default=128, metadata={"help": "FAISS HNSW M value"})
    hnsw_ef_construction: int = field(default=256, metadata={"help": "FAISS HNSW M value"})


def parse_args() -> NegMineArgs:
    parser = HfArgumentParser((NegMineArgs,))
    if len(sys.argv) == 1:
        raise Exception("Expected more than zero args")
    conf_file = sys.argv[1]
    if conf_file.endswith(".yml") or conf_file.endswith(".yaml"):
        config_args_dict = yaml.load(open(conf_file, "r"), Loader=yaml.FullLoader)
        other_args = sys.argv[2:]
        if len(other_args) > 0:
            for key, value in zip(*[iter(other_args)] * 2):
                config_args_dict[key[2:]] = value
        print(config_args_dict)
        (args,) = parser.parse_dict(config_args_dict)
    else:
        (args,) = parser.parse_args_into_dataclasses()
    return args
