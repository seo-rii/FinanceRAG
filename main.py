
from sentence_transformers import CrossEncoder
import logging

from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
from financerag.tasks import FinDER, FinQABench, FinanceBench, TATQA, FinQA, ConvFinQA, MultiHiertt
from FlagEmbedding import FlagLLMReranker
import torch
import glob

# Setup basic logging configuration to show info level messages.
logging.basicConfig(level=logging.INFO)
output_dir = './results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tasks = [FinQA()]
# tasks = [FinQABench(), FinDER(), FinanceBench(), TATQA(), FinQA(), ConvFinQA(), MultiHiertt()]


reranker = None
torch.cuda.empty_cache()

retrieval_model = DenseRetrieval(
    model=SentenceTransformerEncoder(
        model_name_or_path='dunzhang/stella_en_1.5B_v5',
        query_prompt='query: ',
        doc_prompt='passage: ',
        device=device,
        trust_remote_code=True,
    ),
    corpus_chunk_size=100000,
    batch_size=8,
)


retrieval = []

for task in tasks:
    res = task.retrieve(
        retriever=retrieval_model,
    )
    retrieval.append(res)


retrieval_model = None
torch.cuda.empty_cache()

class ReRankerTest(FlagLLMReranker):
    def predict(self, sentences, batch_size):
        return self.compute_score_single_gpu(
            sentence_pairs=sentences, 
            batch_size=batch_size,
        )

reranker = CrossEncoderReranker(model=ReRankerTest('BAAI/bge-reranker-v2-gemma', device=device, trust_remote_code=True, use_bf16=True))


reranking = []

for i in range(len(retrieval)):
    ret = retrieval[i]
    task = tasks[i]
    res = task.rerank(
        reranker=reranker,
        results=ret,
        top_k=300,
        batch_size=4,
    )
    reranking.append(res)
    task.save_results(output_dir=output_dir)


header = 'query_id,corpus_id'

# merge every csv file in the output directory
with open(f'v1.csv', 'w') as f_out:
    f_out.write(header + '\n')
    for file in glob.glob(f'{output_dir}/**/*.csv'):
        with open(file, 'r') as f_in:
            f_in.readline()
            for line in f_in:
                f_out.write(line)