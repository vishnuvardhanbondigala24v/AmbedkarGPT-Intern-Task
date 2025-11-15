# scripts/precompute_embeddings.py
"""
Precompute embeddings for the Ambedkar speech and store them in a persistent Chroma DB.

Usage:
    python scripts/precompute_embeddings.py

Behavior:
- Reads data/speech.txt
- Splits into chunks
- Loads SentenceTransformer model and computes embeddings
- Creates a Chroma PersistentClient and adds documents + embeddings
- Attempts a safe fallback path if the default chroma_db location fails (useful on Windows/OneDrive)
"""

import os
import sys
import tempfile
import traceback
import contextlib
from pathlib import Path

# Disable Chroma telemetry early to avoid extra startup work and potential telemetry errors
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

# Config
DATA_DIR = Path("data")
SPEECH_FILE = DATA_DIR / "speech.txt"
DEFAULT_CHROMA_DIR = Path("chroma_db")
FALLBACK_CHROMA_DIR = Path(tempfile.gettempdir()) / "chroma_db_ambedkar"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "ambedkar"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40

# Helper to load text (use langchain TextLoader if available, otherwise fallback)
def load_text(file_path: Path):
    try:
        from langchain.document_loaders import TextLoader
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        return docs
    except Exception:
        # Fallback: simple read into a single Document-like object
        class SimpleDoc:
            def __init__(self, text):
                self.page_content = text
        text = file_path.read_text(encoding="utf-8")
        return [SimpleDoc(text)]

# Helper to split documents (use langchain splitter if available)
def split_documents(docs):
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(docs)
    except Exception:
        # Very simple fallback splitter: split by paragraphs and then by size
        texts = []
        for d in docs:
            raw = d.page_content
            paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
            for p in paragraphs:
                # naive chunking
                start = 0
                while start < len(p):
                    chunk = p[start:start + CHUNK_SIZE]
                    texts.append(type("D", (), {"page_content": chunk}))
                    start += CHUNK_SIZE - CHUNK_OVERLAP
        return texts

def compute_embeddings(texts, model_name=MODEL_NAME):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("Missing sentence_transformers. Install sentence-transformers and retry.")
        raise e

    print(f"Loading SentenceTransformer model '{model_name}' (this may take a moment)...")
    model = SentenceTransformer(model_name)
    print("Computing embeddings (progress shown by the model)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Convert to plain Python lists for Chroma
    return embeddings.tolist()

def try_create_chroma_client(path: Path):
    """
    Try to create a chromadb PersistentClient at the given path.
    Returns (client, used_path) on success, raises on failure.
    """
    try:
        import chromadb
    except Exception as e:
        print("chromadb is not installed or failed to import. Install chromadb and retry.")
        raise e

    # Ensure parent exists
    path.mkdir(parents=True, exist_ok=True)

    # Suppress chromadb stderr output during client creation
    with open(os.devnull, "w") as _devnull:
        with contextlib.redirect_stderr(_devnull):
            client = chromadb.PersistentClient(path=str(path))
    return client

def main():
    if not SPEECH_FILE.exists():
        print(f"Missing speech file: {SPEECH_FILE}. Please add the speech text and retry.")
        sys.exit(1)

    print("Loading text...")
    docs = load_text(SPEECH_FILE)

    print("Splitting into chunks...")
    split_docs = split_documents(docs)
    texts = [d.page_content for d in split_docs]
    print(f"Created {len(texts)} chunks")

    if len(texts) == 0:
        print("No chunks created; aborting.")
        sys.exit(1)

    # Compute embeddings
    try:
        embeddings = compute_embeddings(texts)
    except Exception as e:
        print("Failed to compute embeddings:")
        traceback.print_exc()
        sys.exit(1)

    # Try to create Chroma client at default path, fallback to temp dir on failure
    chroma_dir = DEFAULT_CHROMA_DIR
    client = None
    try:
        print(f"Creating Chroma DB at: {chroma_dir}")
        client = try_create_chroma_client(chroma_dir)
    except Exception as e:
        print(f"Failed to create Chroma DB at {chroma_dir}: {e}")
        print("Attempting fallback path:", FALLBACK_CHROMA_DIR)
        try:
            chroma_dir = FALLBACK_CHROMA_DIR
            client = try_create_chroma_client(chroma_dir)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("Detailed traceback:")
            traceback.print_exc()
            sys.exit(1)

    # Create or get collection and add documents + embeddings
    try:
        # Suppress stderr during collection operations
        with open(os.devnull, "w") as _devnull:
            with contextlib.redirect_stderr(_devnull):
                collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception:
        # If get_or_create_collection fails, try a direct call without suppression to see error
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    try:
        # If collection already has documents, show count and skip adding
        try:
            existing = collection.count()
        except Exception:
            existing = None

        if existing and existing > 0:
            print(f"Collection '{COLLECTION_NAME}' already has {existing} documents; skipping add.")
        else:
            ids = [f"doc_{i}" for i in range(len(texts))]
            print(f"Adding {len(texts)} documents to collection '{COLLECTION_NAME}'...")
            # Add with embeddings
            with open(os.devnull, "w") as _devnull:
                with contextlib.redirect_stderr(_devnull):
                    collection.add(ids=ids, documents=texts, embeddings=embeddings)
            print("Documents added.")
            try:
                new_count = collection.count()
                print(f"New collection count: {new_count}")
            except Exception:
                pass

        print("Chroma DB ready at:", chroma_dir)
    except Exception as e:
        print("Failed to add documents to Chroma collection:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
