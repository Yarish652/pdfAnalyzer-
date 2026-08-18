from typing import List
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


def sentence_chunker(text: str, sentence_per_chunk: int) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), sentence_per_chunk):
        chunk_sentences = sentences[i:i + sentence_per_chunk]
        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)

    return chunks