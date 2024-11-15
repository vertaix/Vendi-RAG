import os
import sys
import json
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import vendi
sys.path.append("../")


class SentenceTransformerEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, documents):
        # Make the class callable, so it works as an embedding function
        return self.embed_documents(documents)

    def embed_query(self, query):
        return self.model.encode(query, convert_to_tensor=True).tolist()

    def embed_documents(self, documents):
        return self.model.encode(documents, convert_to_tensor=True).tolist()


class Ingestor:
    def __init__(
        self,
        dataset_path: str = "Nan",
        persist_directory: str = "Nan",
    ):
        self.dataset_path = dataset_path
        self.persist_directory = persist_directory
        self.embedding = SentenceTransformerEmbedding("all-MiniLM-L6-v2")

    def load_data(self):
        """Load the dataset and extract documents and metadata."""
        with open(self.dataset_path, "r") as file:
            data = [json.loads(line) for line in file]

        documents = []
        metadatas = []

        for i, sample in enumerate(data):
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

    def create_vectordb(self):
        """Create the Chroma vector store with embeddings and metadata."""
        documents, metadatas = self.load_data()

        print("Initializing Chroma vector store...")
        vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding,  # Use Hugging Face model's encode method
        )

        MAX_BATCH_SIZE = 1000
        total_documents = len(documents)
        print(f"Total documents to process: {total_documents}")

        for start_idx in range(0, total_documents, MAX_BATCH_SIZE):
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_documents)
            batch_texts = documents[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]

            print(
                f"Processing batch {start_idx // MAX_BATCH_SIZE + 1}: "
                f"Documents {start_idx} to {end_idx - 1}"
            )

            vectordb.add_texts(texts=batch_texts, metadatas=batch_metadatas)

        print(f"Persisting the vector store to '{self.persist_directory}'...")
        vectordb.persist()
        print("Vector store created and persisted successfully.")

        return vectordb

    def load_vectordb(self, persist_directory):
        if os.path.exists(persist_directory):
            print(f"Loading existing vector database from '{persist_directory}'...")
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        else:
            print(f"Vector database not found. Creating a new one...")
            vectordb = self.create_vectordb()

        return vectordb

    def query_question(self, question: str, top_n: int = 3, use_mmr: bool = False, use_vendi_score: bool = False, diversity: float = 0.7):
            """
            Query the vector database for a question and retrieve top results.

            Args:
                question (str): The question to search.
                top_n (int): Number of top results to retrieve.
                use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for retrieval.
                use_vendi_score (bool): Whether to use VendiScore for retrieval.
                diversity (float): Diversity parameter for MMR/VendiScore (0 for relevance, 1 for diversity).

            Returns:
                list: List of search results.
            """
            vectordb = self.load_vectordb(self.persist_directory)

            print(f"Searching vector database{' with MMR' if use_mmr else ''}{' with VendiScore' if use_vendi_score else ''} for: {question}")

            if use_mmr or use_vendi_score:
                # Retrieve results with scores
                search_results_with_scores = vectordb.similarity_search_with_score(question, k=top_n * 2)

                # Separate results and scores
                results, scores = zip(*search_results_with_scores)

                # Embed the question and results for similarity computation
                query_embedding = self.embedding.embed_query(question)
                result_embeddings = self.embedding.embed_documents([res.page_content for res in results])

                # Implement MMR or VendiScore
                selected_results = []
                unselected = list(range(len(results)))

                for _ in range(min(top_n, len(results))):
                    if not selected_results:  # First element, pick the highest relevance score
                        idx = unselected.pop(0)
                    else:
                        # Calculate MMR or VendiScore for all remaining candidates
                        scores_list = []
                        for i in unselected:
                            relevance = scores[i]
                            diversity_score = max(
                                1 - cosine(result_embeddings[i], result_embeddings[j])
                                for j in [results.index(sel) for sel in selected_results]
                            )

                            if use_mmr:
                                # MMR score: balance between relevance and diversity
                                score = diversity * relevance - (1 - diversity) * diversity_score
                            elif use_vendi_score:

                                normalized_diversity=vendi.score_X(result_embeddings)
                                score = relevance + diversity * normalized_diversity
                            scores_list.append(score)

                        # Select the candidate with the highest score
                        idx = unselected.pop(scores_list.index(max(scores_list)))

                    selected_results.append(results[idx])

                search_results = selected_results
            else:
                # Standard similarity search
                search_results = vectordb.similarity_search(question, k=top_n)

            if not search_results:
                print("No results found.")
            else:
                print(f"Top {top_n} results for the question:")
                for result in search_results:
                    print(result)

            return search_results

if __name__ == "__main__":
    dataset = "2wikimultihopqa"
    subsample = "test_subsampled"
    top_n = 5
    ingestor = Ingestor(
        dataset_path="../processed_data/{}/{}.jsonl".format(dataset, subsample),
        persist_directory="../vectorDB/{}".format(dataset),
    )
    # vectordb = ingestor.create_vectordb()

    question = "Who is the father-in-law of Queen Hyojeong?"
    results = ingestor.query_question(question, top_n=top_n, use_vendi_score=True, diversity=0.8)
