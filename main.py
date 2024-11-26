# FinDER for example.
# You can use other tasks such as `FinQA`, `TATQA`, etc.
from financerag.common.protocols import CrossEncoder
from financerag.tasks import FinDER
finder_task = FinDER()

from sentence_transformers import SentenceTransformer
from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval
from financerag.rerank import CrossEncoderReranker


model = SentenceTransformer('intfloat/e5-large-v2')
# We need to put prefix for e5 models.
# For more details, see Arxiv paper https://arxiv.org/abs/2212.03533
encoder_model = SentenceTransformerEncoder(
    q_model=model,
    doc_model=model,
    query_prompt='query: ',
    doc_prompt='passage: '
)
retriever = DenseRetrieval(model=encoder_model)

# Retrieve relevant documents
results = finder_task.retrieve(retriever=retriever)

reranker = CrossEncoderReranker(CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2'))
reranked_results = finder_task.rerank(reranker, results, top_k=100, batch_size=32)

finder_task.save_results(output_dir='path_to_save_directory')