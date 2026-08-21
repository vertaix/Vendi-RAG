"""Prompt templates for the Vendi-RAG loop.

These follow the paper's appendix; ``JUDGE_PROMPT`` reproduces the exact
LLM-as-a-judge prompt used in all reported experiments.  Every template asks
for JSON so the outputs can be parsed without a structured-output library.

Override any of them per-instance::

    rag = VendiRAG(retriever, llm, prompts={"judge": MY_JUDGE_PROMPT})
"""

COT_PROMPT = """Generate step-by-step chain-of-thought reasoning to answer the \
question using only the provided documents.

Documents:
{documents}

Question: {question}

Respond with JSON only, in exactly this form:
{{"reasoning": "<your step-by-step reasoning>"}}"""


ANSWER_PROMPT = """Provide a concise and precise answer to the question using \
the retrieved documents and the accumulated reasoning chain. Answer with the \
shortest span that fully answers the question.

Documents:
{documents}

Accumulated reasoning (all iterations):
{reasoning}

Question: {question}

Respond with JSON only, in exactly this form:
{{"answer": "<your concise answer>"}}"""


JUDGE_PROMPT = """You are an expert LLM-based judge tasked with evaluating the \
quality of answers in a Retrieval-Augmented Generation (RAG) system. Your \
evaluation will consider the following aspects:

1. Coherence: Assess whether the provided answer is logically consistent and \
flows smoothly, without conflicting statements or gaps in reasoning.

2. Relevance: Evaluate how well the answer addresses the query based on the \
information from the retrieved documents.

3. Query Alignment: Determine how closely the answer aligns with the specific \
query asked, ensuring that the response is focused and appropriate.

Your evaluation will be quantified based on the following scoring system:
- Coherence Score (C): [1 - 10], where 10 is perfectly coherent.
- Relevance Score (R): [1 - 10], where 10 is highly relevant to the query.
- Query Alignment Score (Q): [1 - 10], where 10 is perfectly aligned.

Query: {question}

Retrieved documents:
{documents}

Answer: {answer}

Respond with JSON only, in exactly this form:
{{"coherence": <int>, "relevance": <int>, "query_alignment": <int>}}"""


REFINE_PROMPT = """The current answer is incomplete or low quality. Reformulate \
the question so that retrieval surfaces the missing information, guided by the \
partial answer and the reasoning chain. Keep the reformulation a single \
self-contained question.

Original question: {question}
Candidate answer: {answer}
Reasoning: {reasoning}

Respond with JSON only, in exactly this form:
{{"refined_question": "<your reformulated question>"}}"""


DEFAULT_PROMPTS = {
    "cot": COT_PROMPT,
    "answer": ANSWER_PROMPT,
    "judge": JUDGE_PROMPT,
    "refine": REFINE_PROMPT,
}
