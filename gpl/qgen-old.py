import os
import re
import json
import argparse
from tqdm.auto import tqdm
import torch
from huggingface_hub import interpreter_login 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

class QGenModel():
    """
    A class that helps with the generation of queries for unstructured and
    unlabelled data. 

    First step of the Gnerative Pseudo Labelling method...
    """
    def __init__(
        self,
        dataset,
        model_path = "BeIR/query-gen-msmarco-t5-base-v1",
        device = None
    ):
        self.pubmed = dataset,
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Creating a folder to store the generated queries.
        if not os.path.exists('generated_data'):
            os.mkdir('generated_data')
        self.data_folder = './generated_data'

        # init the corpus generator object
        #self.init_corpus()

    def cleanup(
        self,
        text,
        pattern=r"[^a-z, 0-9.()]"
    ):
        re_pattern = re.compile(pattern, re.I)
        return re.sub(re_pattern, "", text).replace("  ", " ").replace('\t', ' ').replace('\n', ' ').strip()

    def yield_passage(self):
        for line in self.pubmed:
            yield self.cleanup(line['text'])
    
    def init_corpus(self):
        self.corpus = self.yield_passage()

    def generate(
        self,
        target = 10000,
        max_length = 512,
        batch_size = 32,
        num_queries = 3
    ):
        """
        A method that generates queries from passages using a query generator model of choice.
            
            target: int      -> Number of (query,passage) pairs to be created
            batch_size: int  -> Number of passages to be encoded in a batch, depends on the GPU
            num_queries: int -> Number of queries to generate per passage
        """
        # Begin generation...
        count = 0
        self.passages = []
        passages_batch = []
        self.queries = []
        self.lines = []
        #for i in tqdm(range(0,target,3), desc='generating queries...'):
        with tqdm(total=target) as progress:
            for passage in pubmed:
                passage = self.cleanup(passage['text'])
                if count >= target:break
                passages_batch.append(passage)
                if len(passages_batch) == batch_size:
                
                    # encode the passages
                    encodings = self.tokenizer(
                        passages_batch,
                        truncation=True,
                        padding=True,
                        max_length=max_length,
                        return_tensors='pt'
                    )
                
                    # generate queries for the encodings
                    outputs = self.model.generate(
                        input_ids=encodings['input_ids'].cuda(),
                        attention_mask=encodings['attention_mask'].cuda(),
                        max_length=64,
                        do_sample=True,
                        top_p=0.95,
                        top_k=25,
                        num_return_sequences=num_queries
                    )

                    # decode the generated queries
                    decoded_queries = self.tokenizer.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )

                    for i,query in enumerate(decoded_queries):
                        query = query.replace('\t',' ').replace('\n',' ')
                        pass_idx = int(i/num_queries)
                        query_idx = f"genQ{i}"
                        self.queries.append(query)
                        self.lines.append(query_idx +'\t'+ str(pass_idx) + '\t' + '1')
                        count+=1
                    
                    self.passages.extend(passages_batch)
                    passages_batch = []
                    progress.update(len(decoded_queries))

    def write_query_to_jsonl(self, query_file):
        print("Writing the generated queries to qgen.jsonl")
        queries = [{"_id":f"genQ{i}", "text":q} for i,q in enumerate(self.queries)]
        query_file_path = os.path.join(self.data_folder, query_file)
        with open(query_file_path,'w',encoding='utf-8') as fOut:
            for line in queries:
                json.dump(line,fOut)
                fOut.write('\n')
            
    def write_pass_to_jsonl(self, corpus_file):
        print("Writing the passages to corpus.jsonl file")
        corpus = [{"_id":str(i),"title":'',"text":p} for i,p in enumerate(self.passages)]
        corpus_file_path = os.path.join(self.data_folder, corpus_file)
        with open(corpus_file_path,'w',encoding='utf-8') as fOut:
            for line in corpus:
                json.dump(line,fOut)
                fOut.write('\n')

    def write_qrels_to_tsv(self, qrels_file):
        print("Writting the tab seperated qrels file")
        qrels_file_path = os.path.join(self.data_folder,qrels_file)
        with open(qrels_file_path, 'w', encoding='utf-8') as fOut:
            fOut.write('\n'.join(self.lines))
        
    def write_data(
        self,
        corpus_file = 'corpus.jsonl',
        query_file  = 'queries.jsonl',
        qrels_file  = 'qgen-qrels.tsv'
    ):
        self.write_pass_to_jsonl(corpus_file)
        self.write_query_to_jsonl(query_file)
        self.write_qrels_to_tsv(qrels_file)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='for generating queries')
    parser.add_argument('--target', help='number of queries to be generated', required=True)
    parser.add_argument(
        '--corpus_file_name', 
        help='name of the file in which corpus will be saved',
        required=True
    )
    parser.add_argument(
        '--query_file_name', 
        help='name of the file in which generated queries will be saved',
        required=True
    )
    parser.add_argument(
        '--qrels_file_name', 
        help='name of the file in which generated training data after first step will be saved',
        required=True
    )
    args = parser.parse_args()

    # Pubmed login
    interpreter_login()
    
    # Loaidng the pubmed data
    pubmed = load_dataset("ddp-iitm/pubmed_raw_text_v3", use_auth_token=True, streaming=True, split='train')

    # for line in pubmed:
    #     print(line['text'])
    #     break

    # Initainting a query generating model class
    qgen = QGenModel(dataset=pubmed)

    # Starting the query generatiom
    qgen.generate(target=int(args.target))

    # Writing the data as jsonl files
    qgen.write_data(
        corpus_file = args.corpus_file_name,
        query_file  = args.query_file_name,
        qrels_file  = args.qrels_file_name
    )
