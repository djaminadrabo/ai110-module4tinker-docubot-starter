"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents=None):
        documents = documents if documents is not None else self.load_documents()
        chunks = []
        for filename, content in documents:
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            chunks.extend((filename, paragraph) for paragraph in paragraphs)
        return chunks

    def score_document(self, query: str, document) -> float:
        text = document[1] if isinstance(document, tuple) else document
        query_words = set(query.lower().split())
        doc_words = text.lower().split()
        
        score = 0
        for word in query_words:
            if word in doc_words:
                score += 1
        return score

    def retrieve(self, query: str, top_k: int = 3):
        scored_docs = []
        for doc in self.index:
            score = self.score_document(query, doc)
            if score > 0:
                scored_docs.append((score, doc))

        if not scored_docs:
            return ["I don't know. The documentation does not contain information about this."]

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc[1] if isinstance(doc, tuple) else doc for score, doc in scored_docs[:top_k]]

    def retrieve_snippets(self, query: str, top_k: int = 3):
        scored_docs = []
        for doc in self.index:
            score = self.score_document(query, doc)
            if score > 0:
                scored_docs.append((score, doc))

        if not scored_docs:
            return []

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve_snippets(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
