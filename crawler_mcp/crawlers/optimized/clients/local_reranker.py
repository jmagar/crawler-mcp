from __future__ import annotations

from typing import Any


class LocalReranker:
    """Simple local reranker using PyTorch + Transformers or Sentence-Transformers.

    Prefers sentence-transformers CrossEncoder if available, otherwise falls back
    to raw transformers AutoModelForSequenceClassification with heuristic scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self._backend = None  # "st" or "hf"
        self._model = None
        self._tokenizer = None
        self._resolved_device: str | None = None

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        # Try sentence-transformers CrossEncoder first
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            dev = self.device or ("cuda" if _has_cuda() else "cpu")
            self._model = CrossEncoder(self.model_name, device=dev)
            self._backend = "st"
            self._resolved_device = dev
            return
        except Exception:
            pass
        # Fallback to transformers
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tok = AutoTokenizer.from_pretrained(self.model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(dev)
        mdl.eval()
        self._tokenizer = tok
        self._model = mdl
        self._backend = "hf"
        self._resolved_device = str(dev)

    def rerank(
        self, query: str, documents: list[str], *, top_n: int | None = None
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        self._ensure_backend()
        k = min(len(documents), top_n or len(documents))
        # sentence-transformers CrossEncoder path
        if self._backend == "st":
            from sentence_transformers import CrossEncoder  # type: ignore

            assert isinstance(self._model, CrossEncoder)
            pairs: list[tuple[str, str]] = [(query, d) for d in documents]
            scores = self._model.predict(pairs, convert_to_numpy=True).tolist()
        else:
            # raw transformers path
            import torch  # type: ignore

            assert self._tokenizer is not None
            tok = self._tokenizer
            mdl = self._model
            enc = tok(
                [query] * len(documents),
                documents,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            dev = next(mdl.parameters()).device  # type: ignore
            enc = {k: v.to(dev) for k, v in enc.items()}  # type: ignore
            with torch.inference_mode():
                out = mdl(**enc)
                logits = out.logits
                # Heuristic: if 2-class, use probability of class 1; else use raw score
                if logits.shape[-1] == 2:
                    scores = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
                else:
                    scores = logits.squeeze(-1).detach().cpu().tolist()
        ranked = sorted(enumerate(scores), key=lambda it: it[1], reverse=True)[:k]
        return [{"index": idx, "score": float(score)} for idx, score in ranked]

    def get_device(self) -> str:
        return self._resolved_device or (
            self.device or ("cuda" if _has_cuda() else "cpu")
        )


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:
        return False
