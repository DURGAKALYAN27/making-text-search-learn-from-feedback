import nltk
nltk.download('punkt')

# Our collection of documents
docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
]

print("Sample Documents:")
for i, doc in enumerate(docs):
    print(f"Doc {i}: {doc[:50]}...") # Print first 50 chars
    

import string
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer

# Prepare tools
REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})
TOKENIZER = TreebankWordTokenizer()
STEMMER = PorterStemmer()

# Define the combined function
def tokenize_and_stem(text):
    """Removes punctuation, tokenizes, and stems the text."""
    text_no_punct = text.lower().translate(REMOVE_PUNCTUATION_TABLE)
    tokens = TOKENIZER.tokenize(text_no_punct)
    stems = [STEMMER.stem(token) for token in tokens]
    # Optional: remove very short tokens if desired
    stems = [stem for stem in stems if len(stem) > 1]
    return stems

# Test it on an example document
example_doc = docs[1]
example_doc_tokenized_and_stemmed = tokenize_and_stem(example_doc)

print("\nOriginal Document 1:")
print(example_doc)
print("\nTokenized and Stemmed Document 1:")
print(example_doc_tokenized_and_stemmed)
# Expected output like: ['contact', 'inform', 'email', 'martin', 'davtyan', 'at', 'filament', 'dot', 'ai', 'if', 'you', 'have', 'ani', 'question']


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer using our custom tokenizer/stemmer
# 'stop_words='english'' removes common English words like 'the', 'a', 'is'
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')

# Learn the vocabulary and IDF from the documents
vectorizer.fit(docs)

# Print the learned vocabulary (term -> index mapping)
print("\nLearned Vocabulary:")
print(vectorizer.vocabulary_)
# Expected output like: {'deliver': 11, 'artifici': 2, 'intellig': 17, ...}

# Transform the documents into TF-IDF vectors
doc_vectors = vectorizer.transform(docs)

print("\nShape of TF-IDF Document Matrix:")
print(doc_vectors.shape) # (Number of documents, Number of unique terms)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np # Import numpy

# Define a sample query
query = 'contact email to chat to martin'
print(f"\nSample Query: {query}")

# Transform the query using the *fitted* vectorizer
# Note: transform expects a list/iterable, hence [query]
query_vector = vectorizer.transform([query]) # .todense() is optional for viewing

print("\nQuery Vector (TF-IDF):")
print(query_vector) # This will be a sparse matrix representation

# Calculate cosine similarity between the query vector and all document vectors
similarity_scores = cosine_similarity(query_vector, doc_vectors)

print("\nSimilarity Scores (Query vs Docs):")
print(similarity_scores)
# Expected output like: array([[0.        , 0.48466849, 0.18162735]])
# This means doc 0 has 0 similarity, doc 1 has ~0.48, doc 2 has ~0.18


# Get the indices that would sort the similarities in descending order
# We negate the scores because argsort sorts in ascending order
ranks = (-similarity_scores).argsort(axis=1).flatten() # Flatten to make it a 1D array

print("\nDocument Ranks (Most to Least Relevant):")
print(ranks)
# Expected output like: array([1, 2, 0])

print("\nMost Relevant Document based on TF-IDF:")
most_relevant_doc_index = ranks[0]
print(f"Index: {most_relevant_doc_index}")
print(f"Document: {docs[most_relevant_doc_index]}")
# Expected output: Doc 1 ('Contact information...')


# Structure: {query_string: [(doc_index, relevance_score), ...]}
feedback = {
        'who makes chatbots': [(2, 0.), (0, 1.), (1, 1.), (0, 1.)], # Doc 0, 1 relevant; Doc 2 irrelevant
        'about page': [(0, 1.)] # Doc 0 relevant
}

print("\nSimulated User Feedback:")
print(feedback)


from collections import Counter

# New query we want to score
new_query = 'who is making chatbots information'
print(f"\nNew Query for Feedback Test: {new_query}")

# 1. Find the most similar query in our feedback dictionary
feedback_queries = list(feedback.keys())
if not feedback_queries:
    print("No feedback available yet.")
else:
    # Transform the new query and all feedback queries
    new_query_vector = vectorizer.transform([new_query])
    feedback_query_vectors = vectorizer.transform(feedback_queries)

    # Calculate similarities
    query_similarities = cosine_similarity(new_query_vector, feedback_query_vectors)
    print("\nSimilarity between new query and feedback queries:")
    print(query_similarities)
    # Expected: [[0.70710678, 0.        ]] (More similar to 'who makes chatbots')

    # Find the index and similarity value of the nearest neighbor (NN) query
    nn_idx = np.argmax(query_similarities)
    nn_similarity = np.max(query_similarities)
    nn_query = feedback_queries[nn_idx]

    print(f"\nNearest Feedback Query: '{nn_query}' (Similarity: {nn_similarity:.4f})")

    # 2. Get positive feedback for the NN query
    positive_feedback_entries = [(doc_idx, score) for doc_idx, score
                                 in feedback[nn_query] if score == 1.0]
    pos_feedback_doc_indices = [doc_idx for doc_idx, score in positive_feedback_entries]
    print(f"\nPositive Feedback for '{nn_query}': Docs {pos_feedback_doc_indices}")
    # Expected: Docs [0, 1, 0]

    # 3. Calculate proportions of positive feedback for each document
    if not pos_feedback_doc_indices:
         print("No positive feedback for the nearest query.")
         pos_feedback_proportions = {}
    else:
        counts = Counter(pos_feedback_doc_indices)
        total_positive_feedback = sum(counts.values())
        pos_feedback_proportions = {
            doc_idx: count / total_positive_feedback for doc_idx, count in counts.items()
        }
        print("\nPositive Feedback Proportions for NN Query:")
        print(pos_feedback_proportions)
        # Expected: {0: 0.666..., 1: 0.333...}

    # 4. Create the feature vector: Scale proportions by NN similarity
    # The value for a document is: nn_similarity * (proportion of positive feedback for that doc for the nn_query)
    # If a document had no positive feedback, its proportion is 0.
    pos_feedback_feature_vector = np.array([nn_similarity * pos_feedback_proportions.get(idx, 0.)
                                            for idx, _ in enumerate(docs)])

    print("\nPositive Feedback Feature Vector for the New Query:")
    print(pos_feedback_feature_vector)
    # Expected: [0.4714..., 0.2357..., 0.0       ]
    
    
class Scorer():
    """ Scores documents for a search query based on tf-idf
        similarity and relevance feedback.
    """
    def __init__(self, docs):
        """ Initialize a scorer with a collection of documents, fit a
            vectorizer and list feature functions.
        """
        self.docs = docs
        self.num_docs = len(docs)

        # Use the same preprocessing function
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem,
                                          stop_words='english')
        self.doc_tfidf = self.vectorizer.fit_transform(docs) # Fit and transform docs

        # Feedback storage
        self.feedback = {}
        self._feedback_query_vectors = None # Cache transformed feedback queries

        # Define features and their weights
        # Feature functions should accept a query string and return a numpy array of scores (1 per doc)
        self.features = [
            self._feature_tfidf,
            self._feature_positive_feedback,
            # Add self._feature_negative_feedback here if implementing
        ]
        # Default weights - give feedback slightly more importance initially
        self.feature_weights = [
            1.0, # Weight for TF-IDF
            2.0, # Weight for Positive Feedback
            # Add weight for negative feedback if implementing
        ]

    def score(self, query):
        """ Calculate the final score for a query as a weighted sum of features.
        """
        # Calculate score for each feature
        feature_values = [feature(query) for feature in self.features]

        # Apply weights
        weighted_feature_values = [values * weight for values, weight
                                   in zip(feature_values, self.feature_weights)]

        # Sum the weighted feature values to get the final score per document
        final_scores = np.sum(weighted_feature_values, axis=0)
        return final_scores

    def learn_feedback(self, feedback_dict):
        """ Learn feedback in a form of query -> [(doc index, feedback value)].
            Updates internal feedback storage and pre-calculates vectors.
        """
        self.feedback.update(feedback_dict) # Allow incremental updates
        # Pre-transform feedback queries for efficiency if feedback exists
        if self.feedback:
            feedback_queries = list(self.feedback.keys())
            self._feedback_query_vectors = self.vectorizer.transform(feedback_queries)
        else:
            self._feedback_query_vectors = None

    # --- Feature Calculation Methods ---

    def _feature_tfidf(self, query):
        """ TF-IDF feature: Cosine similarity between query and documents. """
        query_vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel() # .ravel() converts [[score1, score2, ...]] to [score1, score2, ...]

    def _feature_positive_feedback(self, query):
        """ Positive feedback feature: Uses feedback from the nearest neighbor query. """
        # Return zeros if no feedback learned yet or no feedback queries transformed
        if not self.feedback or self._feedback_query_vectors is None:
            return np.zeros(self.num_docs)

        query_vector = self.vectorizer.transform([query])
        feedback_queries = list(self.feedback.keys()) # Ensure order matches vectors

        # Calculate similarities to feedback queries
        query_similarities = cosine_similarity(query_vector, self._feedback_query_vectors)

        # Find nearest neighbor (NN)
        nn_similarity = np.max(query_similarities)
        # Handle case where all similarities might be zero
        if nn_similarity == 0:
            return np.zeros(self.num_docs)

        nn_idx = np.argmax(query_similarities)
        nn_query = feedback_queries[nn_idx]

        # Get positive feedback proportions for the NN query
        pos_feedback_entries = [(doc_idx, score) for doc_idx, score
                                 in self.feedback[nn_query] if score == 1.0]
        pos_feedback_doc_indices = [doc_idx for doc_idx, score in pos_feedback_entries]

        if not pos_feedback_doc_indices:
             return np.zeros(self.num_docs) # No positive feedback for NN query

        counts = Counter(pos_feedback_doc_indices)
        total_positive_feedback = sum(counts.values())
        pos_feedback_proportions = {
            doc_idx: count / total_positive_feedback for doc_idx, count in counts.items()
        }

        # Create feature vector: nn_similarity * proportion
        feature_vector = np.array([nn_similarity * pos_feedback_proportions.get(idx, 0.)
                                   for idx in range(self.num_docs)])
        return feature_vector

    # Optional: Implement negative feedback feature similarly
    # def _feature_negative_feedback(self, query):
    #     # ... similar logic using feedback entries with score == 0.0 ...
    #     # Return a NEGATIVE value for documents with negative feedback
    #     # Example: return -np.array([...])
    #     pass
    
    
# Create the scorer instance
scorer = Scorer(docs)

# Test with the query from Step 9
query = 'who is making chatbots information'
print(f"\n--- Testing Scorer for query: '{query}' ---")

# Get scores *before* learning feedback
# Note: Only TF-IDF feature will contribute significantly now
scores_before_feedback = scorer.score(query)
ranks_before_feedback = (-scores_before_feedback).argsort()

print("\nScores (Before Feedback):")
print(scores_before_feedback)
# Expected: TF-IDF scores dominate, e.g., [0.        , 0.228..., 0.256...]

print("\nRanks (Before Feedback):")
print(ranks_before_feedback)
# Expected: Doc 2 likely highest due to 'chatbot', e.g., [2, 1, 0]

print("\nTop Result (Before Feedback):")
print(docs[ranks_before_feedback[0]])
# Expected: Doc 2 ('Filament Chat...')



# Learn the feedback data we defined earlier
scorer.learn_feedback(feedback)
print("\n--- Feedback Learned ---")

# Get scores *after* learning feedback
scores_after_feedback = scorer.score(query)
ranks_after_feedback = (-scores_after_feedback).argsort()

print("\nScores (After Feedback):")
print(scores_after_feedback)
# Expected: Scores change significantly. The positive feedback feature (weighted * 2)
# for docs 0 and 1 (from the NN query 'who makes chatbots') boosts their scores.
# e.g., [0.94..., 0.69..., 0.25...]

print("\nRanks (After Feedback):")
print(ranks_after_feedback)
# Expected: Doc 0 should now be highest, e.g., [0, 1, 2]

print("\nTop Result (After Feedback):")
print(docs[ranks_after_feedback[0]])
# Expected: Doc 0 ('About us...') - matching the feedback preference!


# Change weights: make TF-IDF more important than feedback
scorer.feature_weights = [0.6, 0.4] # TF-IDF weight = 0.6, Feedback weight = 0.4
print(f"\n--- Changed Feature Weights to: {scorer.feature_weights} ---")

scores_new_weights = scorer.score(query)
ranks_new_weights = (-scores_new_weights).argsort()

print("\nScores (New Weights):")
print(scores_new_weights)
# Expected: Scores will shift again. Maybe doc 1 becomes highest now?
# e.g., [0.188..., 0.231..., 0.154...]

print("\nRanks (New Weights):")
print(ranks_new_weights)
# Expected: Ranking might change again, e.g., [1, 0, 2]

print("\nTop Result (New Weights):")
print(docs[ranks_new_weights[0]])
# Expected: Might revert to Doc 1 ('Contact information...') or stay Doc 0, depending on exact values.