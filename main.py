# FinDER for example.
# You can use other tasks such as `FinQA`, `TATQA`, etc.
from financerag.common.protocols import CrossEncoder
from financerag.tasks import FinDER
from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval
from financerag.rerank import CrossEncoderReranker


finder_task = FinDER()

# We need to put prefix for e5 models.
# For more details, see Arxiv paper https://arxiv.org/abs/2212.03533
encoder_model = SentenceTransformerEncoder(
    query_prompt='query: ',
    doc_prompt='passage: ',
    model_name_or_path='intfloat/e5-large-v2'
)
retriever = DenseRetrieval(model=encoder_model)

# Retrieve relevant documents
results = finder_task.retrieve(retriever=retriever)

reranker = CrossEncoderReranker(CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2'))
reranked_results = finder_task.rerank(reranker, results, top_k=100, batch_size=32)

finder_task.save_results(output_dir='res.jsonl')