import os
import json
import datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import torch, logging

logger = logging.getLogger(__name__)


class QGenModel:
    def __init__(
            self,
            model_path: str,
            gen_prefix: str = "", 
            use_fast: bool = True, 
            device: str = None 
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)

    def gen_query_embeddings(self,examples):

        encodings = self.tokenizer(
            examples['text'],
            padding=True, 
            truncation=True, 
            max_length = 512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outs = self.model.generate(
                input_ids=encodings['input_ids'].to(self.device), 
                do_sample=True,
                max_length=self.max_length,
                top_k=self.top_k,
                top_p=self.top_p, 
                num_return_sequences=self.ques_per_passage
            )

        return {"embeddings": outs}


    def generate(
            self, 
            corpus: datasets.iterable_dataset.IterableDataset,
            num_examples:int,
            output_dir: str, 
            top_p: int = 0.95, 
            top_k: int = 25, 
            max_length: int = 64,
            ques_per_passage: int = 3, 
            prefix: str = "QGen", 
            batch_size: int = 32,
            save: bool = True, 
            save_after: int = 100000
        ):
        self.num_examples = num_examples
        self.output_dir = output_dir
        self.top_p = top_p,
        self.top_k = top_k,
        self.max_length = max_length
        self.ques_per_passage = ques_per_passage
        self.query_prefix = prefix
        
        logger.info("Starting to Generate {} Questions Per Passage using top-p (nucleus) sampling...".format(ques_per_passage))
        logger.info("Params: top_p = {}".format(top_p))
        logger.info("Params: top_k = {}".format(top_k))
        logger.info("Params: max_length = {}".format(max_length))
        logger.info("Params: ques_per_passage = {}".format(ques_per_passage))

        queries = corpus.map(self.gen_query_embeddings, batched=True, remove_columns=['text', 'title'])
        
        # Decoding the queries
        queries = queries.decode(self.decode_queries, batched=True, remove_columns=['_id','embeddings'])

        if save == False:
            return queries

        self.save_in_batches(queries, save_after)


    def decode_queries(self,examples):

        decoded_queries = self.tokenizer.batch_decode(
            examples['embeddings'],
            skip_special_tokens = True
        )
        idx_start = int(examples['_id'])*self.ques_per_passage
        query_ids = [f'{self.query_prefix}{id}' for id in range(idx_start,idx_start+self.ques_per_passage)]
        queries = [{"_id":id, "text":query} for id,query in zip(query_ids,decoded_queries)]

        return {"queries":queries}

    def save_in_batches(
        self,
        queries,
        save_after: int
        ):

        buffer = []
        shard_num = 0
        pbar = tqdm(iter(queries), total = self.num_examples*self.ques_per_passage)

        for example in pbar:
            buffer.append(example)
            if len(buffer) == save_after:
                self.write_to_jsonl(buffer, shard_num)
                buffer = []
                shard_num += 1
        if len(buffer) != 0:
            self.write_to_jsonl(buffer,shard_num)

    def write_to_jsonl(self, buffer, shard_num):

        queries_file = f'queries{shard_num}.jsonl'
        queries_file_path = os.path.join(self.output_dir, queries_file)
        
        logger.info(f"Saving {len(buffer)*self.ques_per_passage} Generated Queries to {queries_file}...")

        with open(queries_file_path, 'w', encoding='utf-8') as fOut:
            for example in buffer:
                queries = example['queries']

                for line in queries:
                    json.dump(line,fOut)
                    fOut.write('\n')

        logger(f'Done writing {queries_file}...')
