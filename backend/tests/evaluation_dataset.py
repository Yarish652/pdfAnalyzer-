"""Gold labels for the local resume retrieval evaluation."""


DOCUMENT_ID = "f0ca67f7-0602-41c2-a65f-ebc4914b8a87"


EVALUATION_QUESTIONS = [
    {
        "question": "What technologies were used to build Speakzy?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 4)},
    },
    {
        "question": "How does Speakzy perform vocabulary revision?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 4)},
    },
    {
        "question": "What role did vector embeddings play in Speakzy?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 4)},
    },
    {
        "question": "What technologies were used in the pronunciation assessment system?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 3)},
    },
    {
        "question": "How does the pronunciation assessment system identify mistakes in speech?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 3)},
    },
    {
        "question": "How did the pronunciation assessment system reduce hallucinations?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 3)},
    },
    {
        "question": "How was the speech analysis pipeline made reliable?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 3)},
    },
    {
        "question": "What backend technologies have you worked with?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 6)},
    },
    {
        "question": "Which projects demonstrate your experience with retrieval augmented generation?",
        "history": [],
        "gold_chunks": {(DOCUMENT_ID, 4), (DOCUMENT_ID, 6)},
    },
    {
        "question": "What AI projects have you built and what was your role in each?",
        "history": [],
        "gold_chunks": {
            (DOCUMENT_ID, 3),
            (DOCUMENT_ID, 4),
            (DOCUMENT_ID, 5),
        },
    },
]