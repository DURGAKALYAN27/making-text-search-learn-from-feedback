# Simple Feedback-Driven Search Engine

A basic Python implementation of a text search engine using TF-IDF and cosine similarity. It includes a mechanism to learn from simple positive user feedback (which results were relevant) to improve search result ranking over time.

## Features

*   Text preprocessing (tokenization, stemming, stopword removal) using NLTK.
*   TF-IDF vectorization using scikit-learn.
*   Document ranking based on cosine similarity between query and documents.
*   Incorporates a "nearest neighbor" positive feedback feature: boosts documents found relevant for similar past queries.
*   Uses a weighted sum of features (TF-IDF similarity, feedback score) for final ranking.

## Setup

1.  **Install Libraries:**
    ```bash
    pip install nltk scikit-learn numpy
    ```
2.  **Download NLTK Data:**
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Quick Usage

```python
from feedback_search import Scorer # Assuming your code is in feedback_search.py

# 1. Sample Data
docs = [
    "This is the first document about AI.",
    "This document discusses machine learning.",
    "A third document, unrelated to AI or ML.",
    "Contact information and AI applications."
]
feedback = {
    'artificial intelligence': [(0, 1.0), (3, 1.0), (1, 0.0)], # Docs 0, 3 relevant
    'info': [(3, 1.0)] # Doc 3 relevant
}

# 2. Initialize Scorer
scorer = Scorer(docs)

# 3. (Optional) Learn from feedback
scorer.learn_feedback(feedback)

# 4. Score a query
query = "information about AI"
scores = scorer.score(query)

# 5. Get ranked document indices (highest score first)
import numpy as np
ranked_indices = np.argsort(-scores) # Negate scores for descending sort

print(f"Query: {query}")
print(f"Scores: {scores}")
print(f"Ranked Doc Indices: {ranked_indices}")
print(f"Top Result: {docs[ranked_indices[0]]}")