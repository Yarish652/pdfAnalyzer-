"""Gold labels for the Deep Learning book chapter 10 (Sequence Modeling) benchmark.

Relevance is defined by evidence phrases rather than chunk indexes so the same
labels stay valid when the extractor or chunker changes. A chunk is relevant
when its normalized text (lowercase, alphanumerics only, NFKC) contains any of
the question's evidence phrases. Keep phrases short so they are unlikely to
straddle a chunk boundary.

Pages are 1-based positions in the PDF, not printed book page numbers.

answer_keywords are used only by the optional generation check: an answer
passes when it contains at least one keyword from every inner list.
"""


EVALUATION_QUESTIONS = [
    {
        "question": "What is the runtime and memory cost of back-propagation through time?",
        "evidence": ["the memory cost is also"],
        "pages": {10},
        "answer_keywords": [["o(τ)", "o(tau)", "linear", "sequence length", "o(t)"]],
    },
    {
        "question": "What is teacher forcing?",
        "evidence": ["receives the ground truth output", "teacher forcing is a training technique"],
        "pages": {11},
        "answer_keywords": [["ground truth", "correct output", "target"]],
    },
    {
        "question": "What is the disadvantage of strict teacher forcing?",
        "evidence": ["disadvantage of strict teacher forcing"],
        "pages": {12},
        "answer_keywords": [["closed-loop", "closed loop", "test time", "fed back", "different"]],
    },
    {
        "question": "Why is an RNN with only output-to-hidden recurrence less powerful than one with hidden-to-hidden recurrence?",
        "evidence": ["strictly less powerful", "less powerful (can express"],
        "pages": {9, 10},
        "answer_keywords": [["output", "o "], ["information", "turing"]],
    },
    {
        "question": "How many units did Siegelmann and Sontag use in their universal RNN?",
        "evidence": ["886 units"],
        "pages": {6},
        "answer_keywords": [["886"]],
    },
    {
        "question": "Which activation function is assumed for the hidden units in the RNN forward propagation equations?",
        "evidence": ["hyperbolic tangent activation function"],
        "pages": {8},
        "answer_keywords": [["tanh", "hyperbolic tangent"]],
    },
    {
        "question": "How does a bidirectional RNN work?",
        "evidence": ["moves backward through time", "both the past and the future", "bidirectional rnns combine"],
        "pages": {22, 23, 24},
        "answer_keywords": [["backward"], ["forward"]],
    },
    {
        "question": "What does an encoder-decoder sequence-to-sequence architecture do?",
        "evidence": ["encoder-decoder or sequence-to-sequence", "reads the input sequence"],
        "pages": {25, 26},
        "answer_keywords": [["encoder"], ["decoder"], ["context", "c "]],
    },
    {
        "question": "What limitation of the encoder-decoder architecture did Bahdanau et al. address?",
        "evidence": ["variable-length sequence rather", "too small to properly summarize"],
        "pages": {26},
        "answer_keywords": [["attention", "variable-length", "variable length", "too small", "fixed-size", "fixed size", "fixed dimension"]],
    },
    {
        "question": "What is the advantage of recursive neural networks over recurrent networks?",
        "evidence": ["can be drastically reduced"],
        "pages": {28},
        "answer_keywords": [["log"]],
    },
    {
        "question": "Why do gradients vanish or explode in recurrent networks?",
        "evidence": ["raised to the power of", "either vanish (most of the time)"],
        "pages": {30, 32},
        "answer_keywords": [["eigenvalue", "multipl", "jacobian", "power"]],
    },
    {
        "question": "What are echo state networks and which weights do they learn?",
        "evidence": ["only learn the output weights", "echo state networks, or esns"],
        "pages": {33},
        "answer_keywords": [["output weights"]],
    },
    {
        "question": "What spectral radius do echo state networks use?",
        "evidence": ["spectral radius such as", "initial spectral radius of 1.2"],
        "pages": {35},
        "answer_keywords": [["3", "1.2"]],
    },
    {
        "question": "What are leaky units?",
        "evidence": ["linear self-connections", "such hidden units are called leaky"],
        "pages": {36, 37},
        "answer_keywords": [["self-connection", "self connection", "running average"]],
    },
    {
        "question": "How do skip connections through time help with long-term dependencies?",
        "evidence": ["time delay of d", "diminish exponentially as a function of", "skip connections through d"],
        "pages": {36},
        "answer_keywords": [["delay", "d time steps", "τ/d", "tau/d", "time steps earlier", "earlier"]],
    },
    {
        "question": "What does the forget gate do in an LSTM?",
        "evidence": ["controlled by a forget gate", "forget gate unit"],
        "pages": {39, 40},
        "answer_keywords": [["self-loop", "self loop", "weight", "forget"]],
    },
    {
        "question": "How does a GRU differ from an LSTM?",
        "evidence": ["single gating unit simultaneously controls"],
        "pages": {41},
        "answer_keywords": [["single gat", "update gate", "reset"]],
    },
    {
        "question": "What bias should be added to the LSTM forget gate?",
        "evidence": ["adding a bias of 1"],
        "pages": {42},
        "answer_keywords": [["1"]],
    },
    {
        "question": "How does gradient norm clipping work?",
        "evidence": ["norm threshold", "clip the norm", "renormalized jointly with a single scaling"],
        "pages": {44},
        "answer_keywords": [["norm"], ["threshold", "rescal", "renormaliz"]],
    },
    {
        "question": "Does gradient clipping help with vanishing gradients?",
        "evidence": ["does not help with vanishing"],
        "pages": {45},
        "answer_keywords": [["not", "no"]],
    },
    {
        "question": "What is a neural Turing machine?",
        "evidence": ["introduced the neural turing", "learn to read from and write arbitrary content"],
        "pages": {46},
        "answer_keywords": [["memory"], ["read", "write"]],
    },
    {
        "question": "What is the difference between content-based and location-based addressing?",
        "evidence": ["location-based addressing is not allowed", "content-based addressing"],
        "pages": {47},
        "answer_keywords": [["content"], ["location", "slot", "address"]],
    },
    {
        "question": "How many parameters does a tabular representation of the joint distribution need compared to an RNN?",
        "evidence": ["tabular representation would"],
        "pages": {17},
        "answer_keywords": [["k", "exponential"]],
    },
]


# Questions the chapter cannot answer. Used only by the generation check,
# which expects the model to refuse.
UNANSWERABLE_QUESTIONS = [
    "What learning rate was used to train GPT-4?",
    "What is the formula for scaled dot-product self-attention in transformers?",
    "Which company sponsored the publication of this book?",
]

REFUSAL_TEXT = "I don't know based on the provided document."
