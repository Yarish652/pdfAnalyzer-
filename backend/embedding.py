import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from backend.config import EMBEDDING_MODEL
except ModuleNotFoundError:
    from config import EMBEDDING_MODEL

# PyTorch defaults to the physical core count; using every logical core was
# 30% faster for embedding on the development laptop, with identical vectors.
torch.set_num_threads(os.cpu_count() or 1)

model = SentenceTransformer(EMBEDDING_MODEL)

# Chunks embedded per step when ingesting; progress is reported per batch.
DOCUMENT_BATCH_SIZE = 16


def embed_texts(texts):
    return model.encode(texts)


def embed_documents(texts, on_progress=None):
    """Embed document chunks in batches, reporting progress after each.

    Texts are processed shortest first so each batch pads to similar lengths,
    then the vectors are returned in the original order.
    """
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    vectors = [None] * len(texts)

    for start in range(0, len(order), DOCUMENT_BATCH_SIZE):
        batch = order[start:start + DOCUMENT_BATCH_SIZE]
        for index, vector in zip(batch, model.encode([texts[index] for index in batch])):
            vectors[index] = vector
        if on_progress:
            on_progress(min(start + DOCUMENT_BATCH_SIZE, len(order)), len(order))

    return np.stack(vectors)
