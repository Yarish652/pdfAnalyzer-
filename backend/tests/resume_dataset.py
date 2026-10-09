"""Evidence-phrase gold labels for the resume regression benchmark.

Same format as long_document_dataset.py. The questions mirror
evaluation_dataset.py, plus education and experience questions that the
chunk-index labels did not cover.
"""


EVALUATION_QUESTIONS = [
    {
        "question": "What technologies were used to build Speakzy?",
        "evidence": ["mongodb atlas, openrouter, gemini embeddings"],
        "pages": {1},
        "answer_keywords": [["react"], ["node"], ["mongodb"]],
    },
    {
        "question": "How does Speakzy perform vocabulary revision?",
        "evidence": ["retrieval-based vocabulary revision"],
        "pages": {1},
        "answer_keywords": [["semantic search", "embedding", "retrieval"]],
    },
    {
        "question": "What role did vector embeddings play in Speakzy?",
        "evidence": ["semantic search and vector embeddings"],
        "pages": {1},
        "answer_keywords": [["vocabulary", "revision", "learned words"]],
    },
    {
        "question": "What technologies were used in the pronunciation assessment system?",
        "evidence": ["whisper, wav2vec2, phonemizer"],
        "pages": {1},
        "answer_keywords": [["whisper"], ["wav2vec2"]],
    },
    {
        "question": "How does the pronunciation assessment system identify mistakes in speech?",
        "evidence": ["configurable rule engine"],
        "pages": {1},
        "answer_keywords": [["rule engine", "substitution", "phoneme"]],
    },
    {
        "question": "How did the pronunciation assessment system reduce hallucinations?",
        "evidence": ["reduce hallucinated feedback"],
        "pages": {1},
        "answer_keywords": [["deterministic", "separat"]],
    },
    {
        "question": "What backend technologies have you worked with?",
        "evidence": ["express.js, fastapi, rest apis"],
        "pages": {1},
        "answer_keywords": [["fastapi", "node", "express"]],
    },
    {
        "question": "What did Speakzy implement for production readiness?",
        "evidence": ["api rate limiting, request validation"],
        "pages": {1},
        "answer_keywords": [["rate limit", "ci", "validation", "testing"]],
    },
    {
        "question": "Where did you do your summer internship and what did you work on?",
        "evidence": ["summer intern", "qr-code based technician login"],
        "pages": {1},
        "answer_keywords": [["honda"]],
    },
    {
        "question": "What is your CGPA in B.Tech?",
        "evidence": ["jamia millia islamia 7/ 10", "jamia millia islamia 7/10"],
        "pages": {1},
        "answer_keywords": [["7"]],
    },
    {
        "question": "What percentage did you score in Class XII?",
        "evidence": ["senior secondary (class xii)"],
        "pages": {1},
        "answer_keywords": [["85"]],
    },
    {
        "question": "Which theory courses have you taken?",
        "evidence": ["computer architecture|digital signal processing", "computer architecture | digital signal processing"],
        "pages": {1},
        "answer_keywords": [["vlsi", "signal", "architecture"]],
    },
]


UNANSWERABLE_QUESTIONS = [
    "What is your expected salary?",
    "Which university did you attend for your master's degree?",
]

REFUSAL_TEXT = "I don't know based on the provided document."
