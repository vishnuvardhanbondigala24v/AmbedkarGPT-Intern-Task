# AmbedkarGPT-Intern-Task

A simple command-line Q&A system built with LangChain + ChromaDB + HuggingFace Embeddings and Ollama (Mistral 7B). The system ingests a short speech by Dr. B.R. Ambedkar and answers questions **based solely on that content**.

## Features
- Load `data/speech.txt`
- Chunk the text
- Embed with `sentence-transformers/all-MiniLM-L6-v2`
- Store and retrieve via local ChromaDB
- Generate answers with Ollama + Mistral 7B (no keys, fully local)

## Requirements
- Python 3.8+
- Ollama installed locally and `mistral` model pulled
- See `requirements.txt` for Python dependencies

## Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
