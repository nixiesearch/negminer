from negminer.args import parse_args
from negminer.log import setup_logging
from accelerate import Accelerator
from negminer.embed.encoder import TextEncoder
from datasets import Dataset
from tqdm import tqdm

logger = setup_logging()

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Negminer embed: {args}")
    accelerator = Accelerator()
    with accelerator.main_process_first():
        corpus = Dataset.load_from_disk(f"{args.out_dir}/corpus").with_format("np")
        queries = Dataset.load_from_disk(f"{args.out_dir}/queries").with_format("np")

    logger.info("Loaded cached dataset")

    encoder = TextEncoder(args, accelerator)
    query_embs = encoder.encode(queries)
    doc_embs = encoder.encode(corpus)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        logger.info("Saving queries")
        qemb_column = [e for e in tqdm(query_embs, desc="unwrapping")]
        qemb = queries.add_column("embed", qemb_column)
        qemb.save_to_disk(f"{args.out_dir}/queries_embed")
        logger.info("Saving corpus")
        cemb_column = [e for e in tqdm(doc_embs, desc="unwrapping")]
        cemb = corpus.add_column("embed", cemb_column)
        cemb.save_to_disk(f"{args.out_dir}/corpus_embed")

    accelerator.end_training()
