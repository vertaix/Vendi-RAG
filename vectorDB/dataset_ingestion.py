import os
import sys

# Must be set before sentence_transformers/transformers are imported to prevent
# TensorFlow (which is installed but not needed) from being pulled in and hanging.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json

# Add the vectorDB directory so `import vendi` resolves regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import vendi
from sklearn.preprocessing import normalize
import heapq
import numpy as np
import logging
from joblib import Parallel, delayed
sys.path.append("../")

Scale_K=5
class SentenceTransformerEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, documents):
        return self.embed_documents(documents)

    def embed_query(self, query):
        return self.model.encode(query, convert_to_tensor=True).tolist()

    def embed_documents(self, documents):
        return self.model.encode(documents, convert_to_tensor=True).tolist()


class Ingestor:
    def __init__(self, dataset_path: str = "Nan", persist_directory: str = "Nan"):
        self.dataset_path = dataset_path
        self.persist_directory = persist_directory
        self.embedding = SentenceTransformerEmbedding("all-mpnet-base-v2")
        self.log_file = os.path.join(persist_directory, "ingestion_log.txt")
        self._vectordb_cache = None   # loaded once, reused across queries
        os.makedirs(persist_directory, exist_ok=True)
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    def load_data(self):
        """Load the dataset and extract documents and metadata."""
        files = [
            # "create_vectordbannotated_only_train.jsonl",
            # "dev_500_subsampled.jsonl",
            # "dev_subsampled.jsonl",
            # "dev.jsonl",
            "test_subsampled.jsonl",
            # "train.jsonl",
        ]
        data = []
        for file_name in files:
            file_path = os.path.join(self.dataset_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    data.extend(json.loads(line) for line in file)
            else:
                logging.warning(f"File {file_path} not found and will be skipped.")
        
        documents, metadatas = [], []
        for sample in data:
            for context in sample["contexts"]:
                documents.append(context["paragraph_text"])
                metadatas.append(
                    {
                        "idx": context["idx"],
                        "title": context.get("title", "Unknown"),
                        "is_supporting": context.get("is_supporting", "N/A"),
                    }
                )

        return documents, metadatas

    def get_completed_batches(self):
        """Read the log file to get the list of completed batches."""
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as log:
                lines = log.readlines()
                completed_batches = {int(line.strip().split(":")[1]) for line in lines}
        else:
            completed_batches = set()
        return completed_batches

    def create_vectordb(self):
        """Create the Chroma vector store with embeddings and metadata."""
        documents, metadatas = self.load_data()

        print("Initializing Chroma vector store...")
        vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding,
        )

        MAX_BATCH_SIZE = 5000
        total_documents = len(documents)

        print(f"Total documents to process: {total_documents}")

        completed_batches = self.get_completed_batches()
        for start_idx in range(0, total_documents, MAX_BATCH_SIZE):
            batch_num = start_idx // MAX_BATCH_SIZE
            if batch_num in completed_batches:
                print(f"Skipping completed batch {batch_num}")
                continue

            end_idx = min(start_idx + MAX_BATCH_SIZE, total_documents)
            batch_texts = documents[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]

            print(f"Processing batch {batch_num + 1}: Documents {start_idx} to {end_idx - 1}")
            vectordb.add_texts(texts=batch_texts, metadatas=batch_metadatas)

            # Log the completion of this batch
            with open(self.log_file, "a") as log:
                log.write(f"Batch:{batch_num}\n")

        print(f"Persisting the vector store to '{self.persist_directory}'...")
        vectordb.persist()
        print("Vector store created and persisted successfully.")

        return vectordb

    def load_vectordb(self, persist_directory):
        if self._vectordb_cache is not None:
            return self._vectordb_cache
        if os.path.exists(persist_directory):
            print(f"Loading existing vector database from '{persist_directory}'...")
            vectordb = Chroma(
                persist_directory=persist_directory, embedding_function=self.embedding
            )
        else:
            print(f"Vector database not found. Creating a new one...")
            vectordb = self.create_vectordb()
        self._vectordb_cache = vectordb
        return vectordb

    def query_question(self, question: str, top_n: int = 3, use_mmr: bool = False, use_vendi_score: bool = False, la: float = 0.7, method: str = ''):
        """
        Query the vector database for a question and retrieve unique results based on `page_content`.

        Args:
            question (str): The question to search.
            top_n (int): Number of top results to retrieve.
            use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for retrieval.
            use_vendi_score (bool): Whether to use VendiScore for retrieval.
            la (float): la parameter for MMR/VendiScore 

        Returns:
            list: List of unique search results based on `page_content`.
        """
        vectordb = self.load_vectordb(self.persist_directory)

        print(f"Searching vector database{' with MMR' if use_mmr else ''}{' with VendiScore' if use_vendi_score else ''} for: {question}")

        search_results = []
        seen_contents = set()  # To track unique `page_content`

        if use_mmr or use_vendi_score:
            # Retrieve results with scores
            search_results_with_scores = vectordb.similarity_search_with_score(question, k=top_n * Scale_K)

            # Separate results and scores
            results, scores = zip(*search_results_with_scores)
            scores = [1/(score+1e-4) for score in list(scores)]
            results = list(results)

            # Embed the question and results for similarity computation
            all_r=[res.page_content for res in results]
            all_r.append(question)
            result_embeddings = self.embedding.embed_documents(all_r)
            normalized_embeddings = normalize(result_embeddings, axis=1)


            if use_mmr:
                for i in range(len(results)):
                    relevance = scores[i]
                    diversity_score = max(
                     cosine(result_embeddings[i], result_embeddings[j])
                    for j in range(len(results)))
                    # MMR score: balance between relevance and diversity
                    score = la * relevance - (1 - la) * diversity_score
                    scores[i] = score

            elif use_vendi_score:
                diversity_scores = self._compute_diversity_scores_parallel(normalized_embeddings, q=.1)
                diversity_scores=np.array(diversity_scores)
                diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min())

                scores=np.array(scores)
                scores = (scores - scores.min()) / (scores.max() - scores.min())
                print("---")
                print(normalized_embeddings.shape)
                print(len(diversity_scores))
                scores = [(10-la)*relevance + la * diversity for relevance, diversity in zip(scores, diversity_scores)]
                # scores = [(10-la)*relevance + la * diversity for relevance, diversity in zip(scores, diversity_scores)]

            # Select the candidate with the highest score
            top_indices = np.argsort(scores)[-top_n * Scale_K:][::-1]

            for idx in top_indices:
                if len(search_results) >= top_n:
                    break
                page_content = results[idx].page_content.strip()  # Use `page_content` for uniqueness
                if page_content not in seen_contents:
                    seen_contents.add(page_content)
                    search_results.append(results[idx])
        else:
            # Standard similarity search
            initial_results = vectordb.similarity_search(question, k=top_n * Scale_K)

            for result in initial_results:
                page_content = result.page_content.strip()  # Use `page_content` for uniqueness
                if page_content not in seen_contents:
                    seen_contents.add(page_content)
                    search_results.append(result)
                    if len(search_results) >= top_n:
                        break

        if not search_results:
            print("No results found.")
        else:
            print(f"Top {top_n} unique results for the question:")

        return search_results

    def load_evaluation_data(self):
        """
        This method loads evaluation data (question + ground-truth) and returns it as a dictionary
        """

        dict_groundtruth = {"question_id": [], "question_text": [], "ground_truth": []}

        with open(self.dataset_path+"test_subsampled.jsonl", "r") as file:
            data = [json.loads(line) for line in file]

        for sample in data:
            dict_groundtruth["question_id"].append(sample["question_id"])
            dict_groundtruth["question_text"].append(sample["question_text"])
            dict_groundtruth["ground_truth"].append(sample["answers_objects"][0]["spans"][0])

        return dict_groundtruth
    def get_candidates(self, question: str, k: int = 50):
        """
        Retrieve top-k candidates via similarity search with L2-normalized embeddings.

        Called by VendiRAG for VRS-based greedy document selection.

        Returns
        -------
        docs : list[Document]
            Candidate documents, ordered by similarity (most similar first).
        doc_embeddings : np.ndarray of shape (n, d), float32, L2-normalized
        query_embedding : np.ndarray of shape (d,), L2-normalized
        """
        vectordb = self.load_vectordb(self.persist_directory)
        docs = vectordb.similarity_search(question, k=k)

        if not docs:
            return [], np.zeros((0, 1), dtype=np.float32), np.zeros(1, dtype=np.float32)

        # Embed all docs + query in one batch, then L2-normalize
        texts = [doc.page_content for doc in docs] + [question]
        raw = np.array(self.embedding.embed_documents(texts), dtype=np.float64)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized = raw / np.maximum(norms, 1e-9)

        doc_embeddings = normalized[:-1]   # (n, d)
        query_embedding = normalized[-1]   # (d,)

        return docs, doc_embeddings, query_embedding

    def _compute_diversity_scores_parallel(self, embeddings, q=1):
        """
        Parallel computation of diversity scores for embeddings.
        """
        def compute_diversity_score(i):
            return vendi.score_X(embeddings[-1].reshape(-1, 1), embeddings[i].reshape(-1, 1), q=q)

        diversity_scores = Parallel(n_jobs=-1)(delayed(compute_diversity_score)(i) for i in range(len(embeddings)))
        return diversity_scores
    

def calculate_vendi_score(retrieved_documents, embedding_function, q=1):
    """
    Calculate the Vendi score for a set of retrieved documents.
    
    Args:
        retrieved_documents (list): List of retrieved document texts.
        embedding_function (callable): A function that generates embeddings for the given documents.
        q (float): The parameter q for Vendi score calculation.
        
    Returns:
        float: The Vendi score for the retrieved documents.
    """
    
    if not retrieved_documents:
        print("No retrieved documents available.")
        return None
    
    # Get embeddings for the retrieved documents
    document_embeddings = embedding_function(retrieved_documents)
    document_embeddings = np.array(document_embeddings)
    document_embeddings = normalize(document_embeddings, axis=1)  # Normalize embeddings
    
    # Compute the Vendi score
    vendi_score = vendi.score_X(document_embeddings, document_embeddings, q=q)
    
    return vendi_score
if __name__ == "__main__":
    dataset = "2wikimultihopqa"

    top_n = 10
    ingestor = Ingestor(
        dataset_path="../processed_data/{}/".format(dataset),
        persist_directory="../vectorDB/{}".format(dataset),
    )
    # vectordb = ingestor.create_vectordb()
    questions = ingestor.load_evaluation_data()['question_text']
    max_query=10
    indexes = np.random.randint(0,500,max_query)
    vendi_rag=[]
    normal_rag=[]
    for ii in indexes:
        question = questions[ii]
        results = ingestor.query_question(question, top_n=top_n, use_vendi_score=True, la=10)
        

        retrieved_texts = [res.page_content for res in results]
        vendi_rag.append(calculate_vendi_score(retrieved_texts, ingestor.embedding.embed_documents))


        results = ingestor.query_question(question, top_n=top_n, use_vendi_score=False, la=10)


        retrieved_texts = [res.page_content for res in results]
        normal_rag.append(calculate_vendi_score(retrieved_texts, ingestor.embedding.embed_documents))
    
    print(f"vendi-rag average vs={np.mean(vendi_rag)}")

    print(f"normal-rag average vs={np.mean(normal_rag)}")

        