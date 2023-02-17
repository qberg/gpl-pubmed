import os
import csv
import json
from tqdm.auto import tqdm

class PubmedDataLoader():
    """
    A class that helps with loading the dataset stored as 
    jsonl and tsv files.
    """
    def __init__(
        self,
        data_path='./generated_data',
        corpus_file='corpus.jsonl',
        queries_file='queries.jsonl',
        qrels_file='qgen-qrels.tsv'
    ):
        self.corpus  = {}
        self.queries = {}
        self.qrels   = {}

        self.corpus_file  = os.path.join(data_path,corpus_file)
        self.queries_file = os.path.join(data_path,queries_file)
        self.qrels_file   = os.path.join(data_path,qrels_file)

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load(self):
        
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.queries_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            print("Loading Corpus...")
            self._load_corpus()
            print(f"Loaded {len(self.corpus)} Documents.")
        
        if not len(self.queries):
            print("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            print(f"Loaded {len(self.queries)} Queries.")

        return self.corpus, self.queries, self.qrels


    def _load_corpus(self):
    
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }
    
    def _load_queries(self):

        num_lines = sum(1 for l in open(self.corpus_file, 'rb'))
        with open(self.queries_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
