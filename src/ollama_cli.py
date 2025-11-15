# src/ollama_cli.py
"""
LangChain-compatible Ollama CLI wrapper.

This implements a small LLM wrapper that calls the local Ollama CLI:
  ollama run <model> "<prompt>"

It subclasses LangChain's LLM base so RetrievalQA and other chains accept it.
"""

from typing import Optional, List, Mapping, Any
import subprocess

from langchain.llms.base import LLM


class OllamaCLI(LLM):
    """Simple LangChain LLM wrapper that calls the Ollama CLI."""

    model: str

    def __init__(self, model: str = "mistral"):
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Core call used by LangChain LLM interface.
        Uses subprocess to call: ollama run <model> "<prompt>"
        Decodes output as UTF-8 with replacement to avoid Windows decode errors.
        """
        cmd = ["ollama", "run", self.model, prompt]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        out = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        err = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.returncode != 0:
            raise RuntimeError(f"Ollama CLI error: {err.strip() or 'unknown error'}")
        return out.strip()

    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "ollama_cli"

    # Convenience so you can call the instance directly: llm(prompt)
    def __call__(self, prompt: str) -> str:
        return self._call(prompt)
