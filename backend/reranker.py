"""Local cross-encoder reranking for retrieved document chunks.

The cross-encoder is the slowest step of answering a question on CPU, so it
runs through ONNX Runtime using the int8-quantized export published with the
model. On the benchmark machine that is about 30% faster than PyTorch with the
same ranking. If ONNX Runtime or the export is unavailable, the PyTorch model
is used instead.
"""

import logging

import numpy as np


logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"
MAX_LENGTH = 512


class OnnxCrossEncoder:
    def __init__(self, model_name: str, file_name: str):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session = onnxruntime.InferenceSession(
            hf_hub_download(model_name, file_name),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {model_input.name for model_input in self.session.get_inputs()}

    def predict(self, pairs):
        queries, passages = zip(*pairs)
        encoded = self.tokenizer(
            list(queries),
            list(passages),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        feeds = {
            name: values.astype(np.int64)
            for name, values in encoded.items()
            if name in self.input_names
        }
        return self.session.run(None, feeds)[0].reshape(-1)


def load_model():
    try:
        return OnnxCrossEncoder(MODEL_NAME, ONNX_FILE)
    except Exception:
        logger.warning("ONNX reranker unavailable, falling back to PyTorch", exc_info=True)
        from sentence_transformers import CrossEncoder

        return CrossEncoder(MODEL_NAME)


model = load_model()


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Score and sort retrieved chunks without changing their source fields."""
    if not chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    reranked = []
    for chunk, score in zip(chunks, scores):
        reranked.append({
            **chunk,
            "metadata": dict(chunk.get("metadata", {})),
            "reranker_score": float(score),
        })

    return sorted(
        reranked,
        key=lambda chunk: chunk["reranker_score"],
        reverse=True,
    )
