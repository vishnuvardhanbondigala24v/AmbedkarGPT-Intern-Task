"""
AmbedkarGPT - Command-line Q&A System

- Uses chromadb PersistentClient + HuggingFaceEmbeddings
- Splits text into chunks for retrieval
- Wraps Chroma in a LangChain retriever
- Uses Ollama (Mistral 7B) for answer generation
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Any

import chromadb
from chromadb.utils import embedding_functions

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document

try:
    from langchain.llms import Ollama  # type: ignore
    _HAS_LANGCHAIN_OLLAMA = True
except Exception:
    _HAS_LANGCHAIN_OLLAMA = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
SPEECH_FILE = DATA_DIR / "speech.txt"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
COLLECTION_NAME = "ambedkar"


class LangChainChromaRetriever(BaseRetriever):
    collection: Any
    k: int = 4

    def _get_relevant_documents(self, query: str) -> List[Document]:
        res = self.collection.query(
            query_texts=[query],
            n_results=self.k,
            include=["documents"],
        )
        docs = res.get("documents", [[]])[0]
        return [Document(page_content=d) for d in docs]


def build_retriever(persist_directory: str = CHROMA_DIR, k: int = 4) -> Tuple[LangChainChromaRetriever, object]:
    if not SPEECH_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SPEECH_FILE}")

    loader = TextLoader(str(SPEECH_FILE), encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    split_docs = splitter.split_documents(docs)

    texts = [d.page_content for d in split_docs]

    client = chromadb.PersistentClient(path=persist_directory)
    chroma_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=chroma_embedding_fn,
    )

    if collection.count() == 0:
        ids = [f"doc_{i}" for i in range(len(texts))]
        collection.add(ids=ids, documents=texts)
        logger.info("Added %d documents to collection '%s'", len(texts), COLLECTION_NAME)

    retriever = LangChainChromaRetriever(collection=collection, k=k)
    return retriever, collection


def main():
    retriever, vectordb = build_retriever()

    use_langchain_qa = False
    qa = None

    if _HAS_LANGCHAIN_OLLAMA:
        try:
            llm = Ollama(model=OLLAMA_MODEL)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            use_langchain_qa = True
        except Exception as e:
            logger.warning("LangChain Ollama failed, fallback to manual RAG: %s", e)

    print("=== AmbedkarGPT Q&A System ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        q = input("Your question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if use_langchain_qa and qa is not None:
            ans = qa.run(q)
        else:
            docs = retriever.get_relevant_documents(q)
            context = "\n\n".join(d.page_content for d in docs)
            prompt = (
                "You are given the following context from a speech. "
                "Answer the question using only the context.\n\n"
                f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
            )
            # Fallback: call Ollama CLI
            import subprocess
            cmd = ["ollama", "run", OLLAMA_MODEL, prompt]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            ans = proc.stdout.strip()

        print(f"\nAnswer: {ans}\n")


if __name__ == "__main__":
    main()
