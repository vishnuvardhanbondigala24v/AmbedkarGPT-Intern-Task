# tests/verify.py
import sys
import pathlib
import shutil

# Ensure repo root is on sys.path so `from src.main import ...` works
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.main import build_retriever

def test_smoke_index_and_retrieve():
    retriever, _ = build_retriever(persist_directory="chroma_db_test", k=2)
    docs = retriever.get_relevant_documents("What is the real remedy?")
    assert len(docs) >= 1
    # cleanup
    shutil.rmtree("chroma_db_test", ignore_errors=True)
