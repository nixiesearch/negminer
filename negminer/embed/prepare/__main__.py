from transformers import AutoTokenizer
from negminer.args import parse_args
from negminer.log import setup_logging
from negminer.embed.prepare.data import Queries
import os

logger = setup_logging()

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"negminer: preprocessing dataset. args: {args}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.embedding)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    corpus = Queries.load_corpus(tokenizer, args.corpus, args.doc_prompt, args.embedding_seq_len)
    corpus.save_to_disk(f"{args.out_dir}/corpus")
    queries = Queries.load_queries(tokenizer, args.queries, args.query_prompt, args.embedding_seq_len)
    queries.save_to_disk(f"{args.out_dir}/queries")
    qrels = Queries.load_qrels(args.qrels, corpus, queries)
    qrels.save_to_disk(f"{args.out_dir}/qrels")

    logger.info("Data preprocessing done")
